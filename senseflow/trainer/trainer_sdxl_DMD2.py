import os
import os.path as osp
import yaml
import random
import numpy as np
import functools
import time
import copy
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, DistributedSampler, default_collate
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    FullStateDictConfig,
    StateDictType,
)
from torch.distributed.fsdp.wrap import (
    size_based_auto_wrap_policy,
)
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from senseflow.utils import instantiate_from_config
from senseflow.trainer.senseflow_utils import extract_into_tensor
from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    UNet2DConditionModel,
)
from transformers import AutoTokenizer, PretrainedConfig
__conditioning_keys__ = {"concat": "c_concat", "crossattn": "c_crossattn", "adm": "y"}
from PIL import Image
from senseflow.data.senseflow_dataset import SDImageDatasetLMDBwoTokenizer
from senseflow.data.senseflow_dataset import cycle

# Adapted from pipelines.StableDiffusionXLPipeline.encode_prompt
def encode_prompt(prompt_batch, text_encoders, tokenizers, proportion_empty_prompts, is_train=True):
    prompt_embeds_list = []

    captions = []
    for caption in prompt_batch:
        if random.random() < proportion_empty_prompts:
            captions.append("")
        elif isinstance(caption, str):
            captions.append(caption)
        elif isinstance(caption, (list, np.ndarray)):
            # take a random caption if there are multiple
            captions.append(random.choice(caption) if is_train else caption[0])

    with torch.no_grad():
        for tokenizer, text_encoder in zip(tokenizers, text_encoders):
            text_inputs = tokenizer(
                captions,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            prompt_embeds = text_encoder(
                text_input_ids.to(text_encoder.device),
                output_hidden_states=True,
            )

            # We are only ALWAYS interested in the pooled output of the final text encoder
            pooled_prompt_embeds = prompt_embeds[0]
            prompt_embeds = prompt_embeds.hidden_states[-2]
            bs_embed, seq_len, _ = prompt_embeds.shape
            prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)
            prompt_embeds_list.append(prompt_embeds)

    prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
    pooled_prompt_embeds = pooled_prompt_embeds.view(bs_embed, -1)
    return prompt_embeds, pooled_prompt_embeds

def compute_embeddings(
        prompt_batch, original_sizes, crop_coords, proportion_empty_prompts, text_encoders, tokenizers, is_train=True
    ):
        target_size = (1024, 1024)
        original_sizes = list(map(list, zip(*original_sizes)))
        crops_coords_top_left = list(map(list, zip(*crop_coords)))

        original_sizes = torch.tensor(original_sizes, dtype=torch.long)
        crops_coords_top_left = torch.tensor(crops_coords_top_left, dtype=torch.long)

        prompt_embeds, pooled_prompt_embeds = encode_prompt(
            prompt_batch, text_encoders, tokenizers, proportion_empty_prompts, is_train
        )
        add_text_embeds = pooled_prompt_embeds

        # Adapted from pipeline.StableDiffusionXLPipeline._get_add_time_ids
        add_time_ids = list(target_size)
        add_time_ids = torch.tensor([add_time_ids])
        add_time_ids = add_time_ids.repeat(len(prompt_batch), 1)
        add_time_ids = torch.cat([original_sizes, crops_coords_top_left, add_time_ids], dim=-1)
        add_time_ids = add_time_ids.cuda().to(dtype=prompt_embeds.dtype)

        prompt_embeds = prompt_embeds.cuda()
        add_text_embeds = add_text_embeds.cuda()
        unet_added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
        return {"prompt_embeds": prompt_embeds, **unet_added_cond_kwargs}
        
def import_model_class_from_model_name_or_path(
        pretrained_model_name_or_path: str, revision: str, subfolder: str = "text_encoder"
    ):
        text_encoder_config = PretrainedConfig.from_pretrained(
            pretrained_model_name_or_path, subfolder=subfolder, revision=revision, use_auth_token=True
        )
        model_class = text_encoder_config.architectures[0]

        if model_class == "CLIPTextModel":
            from transformers import CLIPTextModel

            return CLIPTextModel
        elif model_class == "CLIPTextModelWithProjection":
            from transformers import CLIPTextModelWithProjection

            return CLIPTextModelWithProjection
        else:
            raise ValueError(f"{model_class} is not supported.")


class Trainer(object):

    def __init__(self, config_path, save_path):
        self.config = self.load_config(config_path)
        self.save_path = osp.abspath(save_path)

    def load_config(self, config_path):
        assert osp.exists(config_path), "{} does not exist !".format(config_path)
        with open(config_path) as f:
            config = yaml.safe_load(f)
            f.close()
        return config

    def setup(self):
        self.init_seed()
        self.init_distributed()
        self.init_logger()
        self.init_saver()
        self.init_resume()
        self.build_model()
        self.build_dmd2_dataloader(global_batch_size = self.world_size)
        self.build_optimizer()
        self.build_lr_scheduler()

    def init_resume(self):
        self.resume = False
        self.last_iter = -1
        if not osp.exists(self.save_path) and self.rank == 0:
            os.makedirs(self.save_path)
        last_iter_file = osp.join(self.save_path, "last_iter")
        if not osp.exists(last_iter_file):
            return
        self.resume = True
        with open(last_iter_file) as f:
            self.last_iter = int(f.readlines()[0])
            f.close()
        self.resume_model_ckpt_path = osp.join(
            self.save_path, "ckpt_model_{}.pth".format(self.last_iter)
        )
        self.resume_ema_ckpt_path = osp.join(
            self.save_path, "ckpt_ema_{}.pth".format(self.last_iter)
        )
        self.resume_scheduler_ckpt_path = osp.join(
            self.save_path, "ckpt_scheduler_{}.pth".format(self.last_iter)
        )
        self.last_iter -= 1

    def init_seed(self):
        seed = self.config["train"]["seed"]
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        # random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

    def init_logger(self):
        self.log_interval = self.config["train"]["log_interval"]

    def init_saver(self):
        self.save_interval = self.config["train"]["save_interval"]

    def init_distributed(self):
        self.rank = int(os.environ["RANK"])
        self.world_size = int(os.environ["WORLD_SIZE"])
        self.local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(self.local_rank)
        torch.distributed.init_process_group("nccl")
    
    def build_dmd2_dataloader(
        self,
        global_batch_size: int = 8,
        num_workers: int = 32,
        pin_memory: bool = False,
        persistent_workers: bool = True,
    ):
        dataset_path = self.config["paths"]["dataset"]
        self.real_dataset = SDImageDatasetLMDBwoTokenizer(
                dataset_path=dataset_path
            )
        sampler = DistributedSampler(self.real_dataset, num_replicas=self.world_size, rank=self.local_rank, shuffle=True)
        real_dataloader = DataLoader(
            self.real_dataset,
            batch_size=global_batch_size // self.world_size,
            shuffle=False,  # shuffle is controlled by sampler
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            sampler=sampler,
            collate_fn=default_collate
        )
        self.real_dataloader = cycle(real_dataloader)
        denoising_dataloader = DataLoader(
            self.real_dataset,
            batch_size=global_batch_size // self.world_size,
            shuffle=False,  # shuffle is controlled by sampler
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            sampler=sampler,
            collate_fn=default_collate
        )
        self.denoising_dataloader = cycle(denoising_dataloader)
        self.batch_size = global_batch_size // self.world_size

    def build_model(self):
        # vae
        # text encoder
        # generator / student
        # guidance model
        model_config = self.config.get("model", {})
        
        # from DMD2 code
        self.backward_simulation = False
        self.cls_on_clean_image = model_config.get("cls_on_clean_image", False)
        self.denoising_timestep = model_config.get("denoising_timestep", 1000)
        self.num_denoising_step = model_config.get("num_denoising_step", 4)
        self.num_train_timesteps = 1000
        self.denoising_step_list = torch.tensor(
            list(range(self.denoising_timestep-1, 0, -(self.denoising_timestep//self.num_denoising_step))),
            dtype=torch.long,
        ).cuda()
        self.timestep_interval = self.denoising_timestep//self.num_denoising_step

        # sdxl configs and ckpt paths
        from argparse import Namespace
        args = Namespace()
        args.pretrained_teacher_model = self.config["paths"]["pretrained_model"]
        args.teacher_revision = None
        args.pretrained_vae_model_name_or_path = None 
        args.pretrained_unet_lcm_path = None
        args.num_train_timesteps = 1000
        args.min_step_percent = 0.02
        args.max_step_percent = 0.98
        args.real_guidance_scale = 8.0
        args.fake_guidance_scale = 1.0
        args.diffusion_gan = True
        args.diffusion_gan_max_timestep = 1000

        self.fp16vae = torch.float32
        self.fp16unet = torch.float32

        # load SDXL vae
        vae_path = (
            args.pretrained_teacher_model
            if args.pretrained_vae_model_name_or_path is None
            else args.pretrained_vae_model_name_or_path
        )
        vae = AutoencoderKL.from_pretrained(
            vae_path,
            subfolder="vae" if args.pretrained_vae_model_name_or_path is None else None,
            revision=args.teacher_revision,
            torch_dtype=self.fp16vae,
        )
        vae.enable_gradient_checkpointing()
        
        # load SDXL text encoders
        tokenizer_one = AutoTokenizer.from_pretrained(args.pretrained_teacher_model, subfolder="tokenizer", revision=args.teacher_revision, use_fast=False
        )
        tokenizer_two = AutoTokenizer.from_pretrained(
            args.pretrained_teacher_model, subfolder="tokenizer_2", revision=args.teacher_revision, use_fast=False
        )

        # Load text encoders from SD-XL checkpoint
        # Import correct text encoder classes
        text_encoder_cls_one = import_model_class_from_model_name_or_path(
            args.pretrained_teacher_model, args.teacher_revision
        )
        text_encoder_cls_two = import_model_class_from_model_name_or_path(
            args.pretrained_teacher_model, args.teacher_revision, subfolder="text_encoder_2"
        )

        text_encoder_one = text_encoder_cls_one.from_pretrained(
            args.pretrained_teacher_model, subfolder="text_encoder", revision=args.teacher_revision,
        )
        text_encoder_two = text_encoder_cls_two.from_pretrained(
            args.pretrained_teacher_model, subfolder="text_encoder_2", revision=args.teacher_revision,
        )
        text_encoder_one.requires_grad_(False)
        text_encoder_two.requires_grad_(False)
        text_encoders = [text_encoder_one, text_encoder_two]
        tokenizers = [tokenizer_one, tokenizer_two]
        self.compute_embeddings_fn = functools.partial(
                                                    compute_embeddings,
                                                    text_encoders=text_encoders,
                                                    tokenizers=tokenizers,
                                                    proportion_empty_prompts=0.,
                                                    # caption_column='text',
                                                )

        # load generator
        unet = UNet2DConditionModel.from_pretrained(
                args.pretrained_teacher_model if args.pretrained_unet_lcm_path is None else args.pretrained_unet_lcm_path,
                subfolder="unet" if args.pretrained_unet_lcm_path is None else None,
                revision=args.teacher_revision, torch_dtype=self.fp16unet,
            )
        unet.enable_gradient_checkpointing()

        # guidance model
        fake_unet = UNet2DConditionModel.from_pretrained(
                args.pretrained_teacher_model,
                subfolder="unet",
                revision=args.teacher_revision, torch_dtype=self.fp16unet,
            ).float()
        real_unet = UNet2DConditionModel.from_pretrained(
                args.pretrained_teacher_model,
                subfolder="unet",
            ).float()
        real_unet.requires_grad_(False)
        fake_unet.requires_grad_(True)

        self.guidance_model = GuidanceModel(fake_unet=fake_unet, real_unet=real_unet, args=args)
        try:
            import xformers
            vae.enable_xformers_memory_efficient_attention()
            print('** enable xformer')
        except:
            pass

        self.vae = vae
        self.vae.eval()
        for p in self.vae.parameters():
            p.requires_grad = False
        self.vae.cuda()

        self.scaling_factor = self.vae.config.scaling_factor

        # model = instantiate_from_config(model_config).cuda()
        # pretrain_path = model_config["pretrain"] if "pretrain" in model_config else None
        self.use_hybrid = (
            True if "use_hybrid" not in model_config else model_config["use_hybrid"]
        )

        self.use_full_shard = (
            False
            if "use_full_shard" not in model_config
            else model_config["use_full_shard"]
        )

        if self.use_hybrid and self.use_full_shard:
            raise NotImplementedError

        print("use hybrid: {}".format(self.use_hybrid))

        my_auto_wrap_policy = functools.partial(
            size_based_auto_wrap_policy, min_num_params=0
        )
        print('self.local_rank: ', self.local_rank)
        if self.use_hybrid:
            self.model = FSDP(
                unet.cuda(),
                device_id=self.local_rank,
                sharding_strategy=torch.distributed.fsdp.ShardingStrategy.HYBRID_SHARD,
                auto_wrap_policy=my_auto_wrap_policy,
            )

            self.guidance_model = FSDP(
                self.guidance_model.cuda(),
                device_id=self.local_rank,
                sharding_strategy=torch.distributed.fsdp.ShardingStrategy.HYBRID_SHARD,
                auto_wrap_policy=my_auto_wrap_policy,
            )

        elif self.use_full_shard:
            pass

        else:
            pass

        # noise scheduler
        self.noise_scheduler = DDIMScheduler.from_pretrained(args.pretrained_teacher_model, subfolder="scheduler")
        self.alphas_cumprod = self.noise_scheduler.alphas_cumprod.to('cuda')


    def build_optimizer(self):
        assert "optimizer" in self.config, "config does not contain key 'optimizer'"
        optimizer_config = self.config["optimizer"]

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=1.0,
            betas=(optimizer_config["beta1"], optimizer_config["beta2"]),
            eps=optimizer_config["eps"],
            weight_decay=optimizer_config["weight_decay"],
        )

        self.optimizer_guidance = torch.optim.AdamW(
            [param for param in self.guidance_model.parameters() if param.requires_grad],
            lr=1.0,
            betas=(optimizer_config["beta1"], optimizer_config["beta2"]),
            eps=optimizer_config["eps"],
            weight_decay=optimizer_config["weight_decay"],
        )

    def build_lr_scheduler(self):
        assert (
            "lr_scheduler" in self.config
        ), "config does not contain key 'lr_scheduler'"
        scheduler_config = self.config["lr_scheduler"]
        scheduler = instantiate_from_config(scheduler_config)
        self.lr_scheduler = LambdaLR(self.optimizer, lr_lambda=scheduler.schedule)
        guidance_lr_scheduler = instantiate_from_config(scheduler_config)
        self.guidance_lr_scheduler = LambdaLR(self.optimizer_guidance, lr_lambda=guidance_lr_scheduler.schedule)

    @torch.no_grad()
    def encode_first_stage_model(self, data):
        images = data["image"].cuda()
        batch_size = images.shape[0]
        output = []
        for i in range(batch_size):
            p = self.vae.encode(images[[i]]).latent_dist.sample() * self.scaling_factor
            output.append(p)
        output = torch.cat(output, dim=0)
        return output
    
    @torch.no_grad()
    def sample_backward(self, noisy_image, encoded_text_cond):
        batch_size = noisy_image.shape[0]
        device = noisy_image.device
        prompt_embeds = encoded_text_cond.pop("prompt_embeds")

        # we choose a random step and share it across all gpu
        selected_step = torch.randint(low=0, high=self.num_denoising_step, size=(1,), device=device, dtype=torch.long)
        generated_image = noisy_image  

        for constant in self.denoising_step_list[:selected_step]:
            current_timesteps = torch.ones(batch_size, device=device, dtype=torch.long)  *constant

            generated_noise = self.model(
                noisy_image, current_timesteps, prompt_embeds, added_cond_kwargs=encoded_text_cond
            ).sample

            generated_image = get_x0_from_noise(
                noisy_image, generated_noise.double(), self.alphas_cumprod.double(), current_timesteps
            ).float()

            next_timestep = current_timesteps - self.timestep_interval 
            noisy_image = self.noise_scheduler.add_noise(
                generated_image, torch.randn_like(generated_image), next_timestep
            ).to(noisy_image.dtype)  

        return_timesteps = self.denoising_step_list[selected_step] * torch.ones(batch_size, device=device, dtype=torch.long)
        return generated_image, return_timesteps
    
    @torch.no_grad()
    def prepare_denoising_data(self, denoising_dict, real_train_dict, noise):

        indices = torch.randint(
            0, self.num_denoising_step, (noise.shape[0],), device=noise.device, dtype=torch.long
        )
        timesteps = self.denoising_step_list.to(noise.device)[indices]

        denoising_encoded_text_cond = self.compute_embeddings_fn(denoising_dict['caption'], denoising_dict['orig_size'], denoising_dict['crop_coords'])

        if real_train_dict is not None:
            real_encoded_text_cond = self.compute_embeddings_fn(real_train_dict['caption'], real_train_dict['orig_size'], real_train_dict['crop_coords'])
            real_train_dict['text_embedding'] = real_encoded_text_cond.pop('prompt_embeds')
            real_train_dict['unet_added_conditions'] = real_encoded_text_cond

        if self.backward_simulation:
            # we overwrite the denoising timesteps 
            # note: we also use uncorrelated noise 
            clean_images, timesteps = self.sample_backward(torch.randn_like(noise), copy.deepcopy(denoising_encoded_text_cond))
        else:
            clean_images = denoising_dict['images'].to(noise.device)

        noisy_image = self.noise_scheduler.add_noise(
            clean_images, noise, timesteps
        )

        # set last timestep to pure noise
        pure_noise_mask = (timesteps == (self.num_train_timesteps-1))
        noisy_image[pure_noise_mask] = noise[pure_noise_mask]

        return timesteps, denoising_encoded_text_cond, real_train_dict, noisy_image, clean_images

    def differentiable_decode_first_stage(self, z):
        z = 1.0 / self.scaling_factor * z
        return self.vae.decode(z).sample

    def scalings_for_boundary_conditions(self, timestep, sigma_data=0.5, timestep_scaling=10.0):
        c_skip = sigma_data**2 / ((timestep / 0.1) ** 2 + sigma_data**2)
        c_out = (timestep / 0.1) / ((timestep / 0.1) ** 2 + sigma_data**2) ** 0.5
        return c_skip, c_out

    def append_dims(self, x, target_dims):
        """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
        dims_to_append = target_dims - x.ndim
        if dims_to_append < 0:
            raise ValueError(f"input has {x.ndim} dims but target_dims is {target_dims}, which is less")
        return x[(...,) + (None,) * dims_to_append]

    def train(self):
        self.model.train()
        self.guidance_model.train()
        scaler = ShardedGradScaler()
        guidance_scaler = ShardedGradScaler()
        iter_time_list = []
        torch.cuda.synchronize()
        dist.barrier()
        iter_begin_time = time.time()

        self.num_total_iters = 10000000
        self.GD = 5  # generator and discriminator update ratio
        self.cls_on_clean_image = True
        self.guidance_cls_loss_weight = 0.01
        self.gen_cls_loss_weight = 0.005
        self.gan_alone = False
        self.gen_cls_loss = True

        self.latent_channel = 4
        self.latent_resolution = 128
        self.network_context_manager = torch.autocast(device_type="cuda", dtype=torch.bfloat16)

        rank, world_size = dist.get_rank(), dist.get_world_size()
        current_iter = 0
        for index in range(self.num_total_iters):
            current_iter += 1                
            COMPUTE_GENERATOR_GRADIENT = current_iter % self.GD == 0
            denoising_dict = next(self.denoising_dataloader)
            real_train_dict = next(self.real_dataloader)

            noise = torch.randn(self.batch_size, self.latent_channel, self.latent_resolution, self.latent_resolution).cuda()

            # generator forward
            timesteps, denoising_encoded_text_cond, real_train_dict, noisy_image, clean_image = self.prepare_denoising_data(
                    denoising_dict, real_train_dict, noise
                )
            prompt_embeds = denoising_encoded_text_cond.pop("prompt_embeds")
            # Build uncond
            uncond = copy.deepcopy(denoising_encoded_text_cond)
            uncond["text_embeds"] = torch.zeros_like(denoising_encoded_text_cond["text_embeds"]).cuda()
            uncond_prompt_embeds = torch.zeros_like(prompt_embeds).cuda().float()
            if COMPUTE_GENERATOR_GRADIENT:
                with self.network_context_manager:
                    generated_noise = self.model(
                        noisy_image, timesteps.long(), 
                        prompt_embeds, added_cond_kwargs=denoising_encoded_text_cond
                    ).sample
            else:
                with torch.no_grad():
                    generated_noise = self.model(
                        noisy_image, timesteps.long(), 
                        prompt_embeds, added_cond_kwargs=denoising_encoded_text_cond
                    ).sample
            generated_image = get_x0_from_noise(
                noisy_image.double(), 
                generated_noise.double(), self.alphas_cumprod.double(), timesteps
            ).float()

            if COMPUTE_GENERATOR_GRADIENT:
                generator_data_dict = {
                    "image": generated_image,
                    "text_embedding": prompt_embeds,
                    "uncond_embedding": uncond_prompt_embeds,
                    "real_train_dict": real_train_dict,
                    "unet_added_conditions": denoising_encoded_text_cond,
                    "uncond_unet_added_conditions": uncond
                } 

                # avoid any side effects of gradient accumulation
                self.guidance_model.requires_grad_(False)
                generator_loss_dict, generator_log_dict = self.guidance_model(
                    generator_turn=True,
                    guidance_turn=False,
                    generator_data_dict=generator_data_dict
                )
            else:
                generator_loss_dict = {}
                generator_log_dict = {}

            generator_log_dict["guidance_data_dict"] = {
                "image": generated_image.detach(),
                "text_embedding": prompt_embeds.detach(),
                "uncond_embedding": uncond_prompt_embeds.detach(),
                "real_train_dict": real_train_dict,
                "unet_added_conditions": denoising_encoded_text_cond,
                "uncond_unet_added_conditions": uncond
            }
            generator_log_dict['denoising_timestep'] = timesteps

            # first update the generator if the current step is a multiple of dfake_gen_update_ratio
            generator_loss = 0.0 
            if COMPUTE_GENERATOR_GRADIENT:
                if not self.gan_alone:
                    generator_loss += generator_loss_dict["loss_dm"]

                if self.cls_on_clean_image and self.gen_cls_loss:
                    generator_loss += generator_loss_dict["gen_cls_loss"] * self.gen_cls_loss_weight
                    
                scaler.scale(generator_loss).backward()
                scaler.unscale_(self.optimizer)
                self.model.clip_grad_norm_(1.0)
                scaler.step(self.optimizer)
                scaler.update()
                self.optimizer.zero_grad()
                self.optimizer_guidance.zero_grad()

            self.lr_scheduler.step()


            # update guidance model (dfake and classifier)
            self.guidance_model.requires_grad_(True)
            guidance_loss_dict, guidance_log_dict = self.guidance_model(
                generator_turn=False,
                guidance_turn=True,
                guidance_data_dict=generator_log_dict['guidance_data_dict']
            )
            guidance_loss = 0 

            guidance_loss += guidance_loss_dict["loss_fake_mean"]

            if self.cls_on_clean_image:
                guidance_loss += guidance_loss_dict["guidance_cls_loss"] * self.guidance_cls_loss_weight
            
            guidance_scaler.scale(guidance_loss).backward()
            guidance_scaler.unscale_(self.optimizer_guidance)
            self.guidance_model.clip_grad_norm_(1.0)
            guidance_scaler.step(self.optimizer_guidance)
            guidance_scaler.update()
            self.optimizer.zero_grad()
            self.optimizer_guidance.zero_grad()
            self.guidance_lr_scheduler.step()

            torch.cuda.synchronize()
            dist.barrier()
            iter_end_time = time.time()
            iter_time = iter_end_time - iter_begin_time
            iter_begin_time = iter_end_time
            iter_time_list.append(iter_time)
            if len(iter_time_list) > 100:
                _ = iter_time_list.pop(0)

            # save tensors
            self.image_logger_save_dir = self.save_path
            if (current_iter % 25 == 0) and self.rank == 0:
                with torch.no_grad():
                    batchid = 0
                    model_pred_img = self.differentiable_decode_first_stage(generated_image.float())
                    clean_img = self.differentiable_decode_first_stage(clean_image.float())
                    x_start_img = self.differentiable_decode_first_stage(denoising_dict['images'].float().cuda())
                    if COMPUTE_GENERATOR_GRADIENT:
                        dmtrain_pred_real_image = self.differentiable_decode_first_stage(generator_log_dict['dmtrain_pred_real_image'].float().cuda())
                        dmtrain_pred_fake_image = self.differentiable_decode_first_stage(generator_log_dict['dmtrain_pred_fake_image'].float().cuda())
                    faketrain_x0_pred =  self.differentiable_decode_first_stage(guidance_log_dict['faketrain_x0_pred'].float())
                    print(f"save image {denoising_dict['caption'][batchid]}")
                    if COMPUTE_GENERATOR_GRADIENT:
                        save_model_pred_img , save_clean_img, save_x_start_img, save_dmtrain_pred_real_image, save_dmtrain_pred_fake_image, save_faketrain_x0_pred = \
                            model_pred_img[[batchid]], clean_img[[batchid]], x_start_img[[batchid]], dmtrain_pred_real_image[[batchid]], dmtrain_pred_fake_image[[batchid]], faketrain_x0_pred[[batchid]]
                    else:
                        save_model_pred_img , save_clean_img, save_x_start_img, save_faketrain_x0_pred = \
                            model_pred_img[[batchid]], clean_img[[batchid]], x_start_img[[batchid]], faketrain_x0_pred[[batchid]]
                    th2np = lambda x: x[batchid].detach().permute(1, 2, 0).cpu().numpy()
                    np2save = lambda x: ((x + 1).clip(0, 2) / 2. * 255).astype(np.uint8)
                    res = []
                    if COMPUTE_GENERATOR_GRADIENT:
                        for item in [save_model_pred_img, save_clean_img, save_x_start_img, save_dmtrain_pred_real_image, save_dmtrain_pred_fake_image, save_faketrain_x0_pred]:
                            res.append(np2save(th2np(item)))
                    else:
                        for item in [save_model_pred_img, save_clean_img, save_x_start_img, save_faketrain_x0_pred]:
                            res.append(np2save(th2np(item)))
                    os.makedirs(os.path.join(self.image_logger_save_dir, 'image_log'), exist_ok=True)
                    tpath = os.path.join(self.image_logger_save_dir, 'image_log', 'image_log-%d-%d.jpg' % (current_iter, generator_log_dict['denoising_timestep'][0].cpu().item()))
                    print('log img save', tpath)
                    res = np.concatenate(res, axis=1)
                    Image.fromarray(res).save(tpath)

            if current_iter % 25 == 0 and self.rank == 0:
                if COMPUTE_GENERATOR_GRADIENT:
                    print(
                        "iter {}/{}, generator_loss: {}, loss_dm: {}, gen_cls_loss: {}, guidance_loss: {}, loss_fake_mean: {}, guidance_cls_loss: {}, lr: {}, guidance_lr: {}, iter time avg: {}, iter time: {}".format(
                            current_iter,
                            self.num_total_iters,
                            generator_loss,
                            generator_loss_dict["loss_dm"],
                            generator_loss_dict["gen_cls_loss"],
                            guidance_loss,
                            guidance_loss_dict["loss_fake_mean"],
                            guidance_loss_dict["guidance_cls_loss"],
                            self.optimizer.param_groups[0]["lr"],
                            self.optimizer_guidance.param_groups[0]["lr"],
                            sum(iter_time_list) / len(iter_time_list),
                            iter_time,
                        )
                    )
                else:
                    print(
                        "iter {}/{}, generator_loss: {}, guidance_loss: {}, loss_fake_mean: {}, guidance_cls_loss: {}, lr: {}, guidance_lr: {}, iter time avg: {}, iter time: {}".format(
                            current_iter,
                            self.num_total_iters,
                            generator_loss,
                            guidance_loss,
                            guidance_loss_dict["loss_fake_mean"],
                            guidance_loss_dict["guidance_cls_loss"],
                            self.optimizer.param_groups[0]["lr"],
                            self.optimizer_guidance.param_groups[0]["lr"],
                            sum(iter_time_list) / len(iter_time_list),
                            iter_time,
                        )
                    )

            if (current_iter + 1) % self.save_interval == 0:
                self.save(current_iter + 1)

    def get_current_state_dict(self):
        fullstate_save_policy = FullStateDictConfig(
            offload_to_cpu=True, rank0_only=True
        )

        with FSDP.state_dict_type(
            self.model, StateDictType.FULL_STATE_DICT, fullstate_save_policy
        ):
            model_state = self.model.state_dict()
        return model_state

    def save(self, save_iter):
        fullstate_save_policy = FullStateDictConfig(
            offload_to_cpu=True, rank0_only=True
        )

        with FSDP.state_dict_type(
            self.model, StateDictType.FULL_STATE_DICT, fullstate_save_policy
        ):
            model_state = self.model.state_dict()
        with FSDP.state_dict_type(
            self.guidance_model, StateDictType.FULL_STATE_DICT, fullstate_save_policy
        ):
            guidance_model_state = self.guidance_model.state_dict()

        if self.rank == 0:

            model = {"model": model_state}
            guidance = {"guidance_model": guidance_model_state}

            model_save_path = osp.join(
                self.save_path, "ckpt_model_{}.pth".format(save_iter)
            )
            guidance_save_path = osp.join(
                self.save_path, "ckpt_guidance_{}.pth".format(save_iter)
            )

            print("saving checkpoint to {}".format(model_save_path))
            torch.save(model, model_save_path)
            torch.save(guidance, guidance_save_path)

            with open(osp.join(self.save_path, "last_iter"), "w+") as f:
                f.writelines([str(save_iter)])
                f.close()
            print("save done")

        torch.cuda.synchronize()
        dist.barrier()

from senseflow.trainer.senseflow_utils import get_x0_from_noise, predict_noise
from senseflow.trainer.dmd_sd_unet_forward import classify_forward
import types 
import torch.nn as nn

class GuidanceModel(nn.Module):
    def __init__(self, fake_unet, real_unet, args):
        super().__init__()
        self.fake_unet = fake_unet
        self.real_unet = real_unet
        # Set real net precision to bf16
        self.real_unet = self.real_unet.to(torch.bfloat16)
        self.scheduler = DDIMScheduler.from_pretrained(
            args.pretrained_teacher_model,
            subfolder="scheduler"
        )
        alphas_cumprod = self.scheduler.alphas_cumprod
        self.register_buffer(
            "alphas_cumprod",
            alphas_cumprod
        )

        self.num_train_timesteps = args.num_train_timesteps
        self.min_step = int(args.min_step_percent * self.scheduler.num_train_timesteps)
        self.max_step = int(args.max_step_percent * self.scheduler.num_train_timesteps)
        
        self.real_guidance_scale = args.real_guidance_scale 
        self.fake_guidance_scale = args.fake_guidance_scale
        self.diffusion_gan = args.diffusion_gan
        self.diffusion_gan_max_timestep = args.diffusion_gan_max_timestep
        self.use_fp16 = True
        self.sdxl = True

        self.network_context_manager = torch.autocast(device_type="cuda", dtype=torch.bfloat16)

        # dmd2 gan
        self.cls_on_clean_image = True
        self.gan_alone = False

        if self.cls_on_clean_image:
            self.fake_unet.forward = types.MethodType(
                classify_forward, self.fake_unet
            )
            self.cls_pred_branch = nn.Sequential(
                nn.Conv2d(kernel_size=4, in_channels=1280, out_channels=1280, stride=2, padding=1), # 32x32 -> 16x16 
                nn.GroupNorm(num_groups=32, num_channels=1280),
                nn.SiLU(),
                nn.Conv2d(kernel_size=4, in_channels=1280, out_channels=1280, stride=2, padding=1), # 16x16 -> 8x8 
                nn.GroupNorm(num_groups=32, num_channels=1280),
                nn.SiLU(),
                nn.Conv2d(kernel_size=4, in_channels=1280, out_channels=1280, stride=2, padding=1), # 8x8 -> 4x4
                nn.GroupNorm(num_groups=32, num_channels=1280),
                nn.SiLU(),
                nn.Conv2d(kernel_size=4, in_channels=1280, out_channels=1280, stride=4, padding=0), # 4x4 -> 1x1
                nn.GroupNorm(num_groups=32, num_channels=1280),
                nn.SiLU(),
                nn.Conv2d(kernel_size=1, in_channels=1280, out_channels=1, stride=1, padding=0), # 1x1 -> 1x1
            )
    
    def compute_distribution_matching_loss(
        self, 
        latents,
        text_embedding,
        uncond_embedding,
        unet_added_conditions,
        uncond_unet_added_conditions
    ):
        original_latents = latents 
        batch_size = latents.shape[0]
        with torch.no_grad():
            timesteps = torch.randint(
                self.min_step, 
                min(self.max_step+1, self.num_train_timesteps),
                [batch_size], 
                device=latents.device,
                dtype=torch.long
            )

            noise = torch.randn_like(latents)

            noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)

            # run at full precision as autocast and no_grad doesn't work well together 
            pred_fake_noise = predict_noise(
                self.fake_unet, noisy_latents, text_embedding, uncond_embedding, 
                timesteps, guidance_scale=self.fake_guidance_scale,
                unet_added_conditions=unet_added_conditions,
                uncond_unet_added_conditions=uncond_unet_added_conditions
            )  

            pred_fake_image = get_x0_from_noise(
                noisy_latents.double(), pred_fake_noise.double(), self.alphas_cumprod.double(), timesteps
            )

            if self.use_fp16:
                if self.sdxl:
                    bf16_unet_added_conditions = {} 
                    bf16_uncond_unet_added_conditions = {} 

                    for k,v in unet_added_conditions.items():
                        bf16_unet_added_conditions[k] = v.to(torch.bfloat16)
                    for k,v in uncond_unet_added_conditions.items():
                        bf16_uncond_unet_added_conditions[k] = v.to(torch.bfloat16)
                else:
                    bf16_unet_added_conditions = unet_added_conditions 
                    bf16_uncond_unet_added_conditions = uncond_unet_added_conditions

                pred_real_noise = predict_noise(
                    self.real_unet, noisy_latents.to(torch.bfloat16), text_embedding.to(torch.bfloat16), 
                    uncond_embedding.to(torch.bfloat16), 
                    timesteps, guidance_scale=self.real_guidance_scale,
                    unet_added_conditions=bf16_unet_added_conditions,
                    uncond_unet_added_conditions=bf16_uncond_unet_added_conditions
                ) 
            else:
                pred_real_noise = predict_noise(
                    self.real_unet, noisy_latents, text_embedding, uncond_embedding, 
                    timesteps, guidance_scale=self.real_guidance_scale,
                    unet_added_conditions=unet_added_conditions,
                    uncond_unet_added_conditions=uncond_unet_added_conditions
                )

            pred_real_image = get_x0_from_noise(
                noisy_latents.double(), pred_real_noise.double(), self.alphas_cumprod.double(), timesteps
            )     

            p_real = (latents - pred_real_image)
            p_fake = (latents - pred_fake_image)

            grad = (p_real - p_fake) / torch.abs(p_real).mean(dim=[1, 2, 3], keepdim=True) 
            grad = torch.nan_to_num(grad)

        loss = 0.5 * F.mse_loss(original_latents.float(), (original_latents-grad).detach().float(), reduction="mean")         

        loss_dict = {
            "loss_dm": loss 
        }

        dm_log_dict = {
            "dmtrain_noisy_latents": noisy_latents.detach().float(),
            "dmtrain_pred_real_image": pred_real_image.detach().float(),
            "dmtrain_pred_fake_image": pred_fake_image.detach().float(),
            "dmtrain_grad": grad.detach().float(),
            "dmtrain_gradient_norm": torch.norm(grad).item()
        }

        return loss_dict, dm_log_dict
    
    def compute_loss_fake(
        self,
        latents,
        text_embedding,
        uncond_embedding,
        unet_added_conditions=None,
        uncond_unet_added_conditions=None
    ):
        latents = latents.detach()
        batch_size = latents.shape[0]
        noise = torch.randn_like(latents)

        timesteps = torch.randint(
            0,
            self.num_train_timesteps,
            [batch_size], 
            device=latents.device,
            dtype=torch.long
        )
        noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)

        with self.network_context_manager:
            fake_noise_pred = predict_noise(
                self.fake_unet, noisy_latents, text_embedding, uncond_embedding,
                timesteps, guidance_scale=1, # no guidance for training dfake 
                unet_added_conditions=unet_added_conditions,
                uncond_unet_added_conditions=uncond_unet_added_conditions
            )

        fake_noise_pred = fake_noise_pred.float()

        fake_x0_pred = get_x0_from_noise(
            noisy_latents.double(), fake_noise_pred.double(), self.alphas_cumprod.double(), timesteps
        )

        # epsilon prediction loss 
        loss_fake = torch.mean(
            (fake_noise_pred.float() - noise.float())**2
        )

        loss_dict = {
            "loss_fake_mean": loss_fake,
        }

        fake_log_dict = {
            "faketrain_latents": latents.detach().float(),
            "faketrain_noisy_latents": noisy_latents.detach().float(),
            "faketrain_x0_pred": fake_x0_pred.detach().float()
        }
        return loss_dict, fake_log_dict

    def compute_generator_clean_cls_loss(self, 
        fake_image, text_embedding, 
        unet_added_conditions=None
    ):
        loss_dict = {} 

        pred_realism_on_fake_with_grad = self.compute_cls_logits(
            fake_image, 
            text_embedding=text_embedding, 
            unet_added_conditions=unet_added_conditions
        )
        loss_dict["gen_cls_loss"] = F.softplus(-pred_realism_on_fake_with_grad).mean()
        return loss_dict 

    def generator_forward(
        self,
        image,
        text_embedding,
        uncond_embedding,
        unet_added_conditions=None,
        uncond_unet_added_conditions=None
    ):
        loss_dict = {}
        log_dict = {}

        # image.requires_grad_(True)
        if not self.gan_alone:
            dm_dict, dm_log_dict = self.compute_distribution_matching_loss(
                image, text_embedding, uncond_embedding, 
                unet_added_conditions, uncond_unet_added_conditions
            )

        loss_dict.update(dm_dict)
        log_dict.update(dm_log_dict)

        if self.cls_on_clean_image:
            clean_cls_loss_dict = self.compute_generator_clean_cls_loss(
                image, text_embedding, unet_added_conditions
            )
            loss_dict.update(clean_cls_loss_dict)

        return loss_dict, log_dict 

    def compute_cls_logits(self, image, text_embedding, unet_added_conditions):
        # we are operating on the VAE latent space, no further normalization needed for now 
        if self.diffusion_gan:
            timesteps = torch.randint(
                0, self.diffusion_gan_max_timestep, [image.shape[0]], device=image.device, dtype=torch.long
            )
            image = self.scheduler.add_noise(image, torch.randn_like(image), timesteps)
        else:
            timesteps = torch.zeros([image.shape[0]], dtype=torch.long, device=image.device)

        with self.network_context_manager:
            rep = self.fake_unet.forward(
                image, timesteps, text_embedding,
                added_cond_kwargs=unet_added_conditions,
                classify_mode=True
            )

        # we only use the bottleneck layer 
        rep = rep[-1].float()
        logits = self.cls_pred_branch(rep).squeeze(dim=[2, 3])
        return logits

    def compute_guidance_clean_cls_loss(
            self, real_image, fake_image, 
            real_text_embedding, fake_text_embedding,
            real_unet_added_conditions=None, 
            fake_unet_added_conditions=None
        ):
        pred_realism_on_real = self.compute_cls_logits(
            real_image.detach(), 
            text_embedding=real_text_embedding,
            unet_added_conditions=real_unet_added_conditions
        )
        pred_realism_on_fake = self.compute_cls_logits(
            fake_image.detach(), 
            text_embedding=fake_text_embedding,
            unet_added_conditions=fake_unet_added_conditions
        )

        log_dict = {
            "pred_realism_on_real": torch.sigmoid(pred_realism_on_real).squeeze(dim=1).detach(),
            "pred_realism_on_fake": torch.sigmoid(pred_realism_on_fake).squeeze(dim=1).detach()
        }

        classification_loss = F.softplus(pred_realism_on_fake).mean() + F.softplus(-pred_realism_on_real).mean()
        loss_dict = {
            "guidance_cls_loss": classification_loss
        }
        return loss_dict, log_dict 
    
    def guidance_forward(
        self,
        image,
        text_embedding,
        uncond_embedding,
        real_train_dict=None,
        unet_added_conditions=None,
        uncond_unet_added_conditions=None
    ):
        fake_dict, fake_log_dict = self.compute_loss_fake(
            image, text_embedding, uncond_embedding,
            unet_added_conditions=unet_added_conditions,
            uncond_unet_added_conditions=uncond_unet_added_conditions
        )

        loss_dict = fake_dict 
        log_dict = fake_log_dict

        if self.cls_on_clean_image:
            clean_cls_loss_dict, clean_cls_log_dict = self.compute_guidance_clean_cls_loss(
                real_image=real_train_dict['images'], 
                fake_image=image,
                real_text_embedding=real_train_dict['text_embedding'],
                fake_text_embedding=text_embedding, 
                real_unet_added_conditions=real_train_dict['unet_added_conditions'],
                fake_unet_added_conditions=unet_added_conditions
            )
            loss_dict.update(clean_cls_loss_dict)
            log_dict.update(clean_cls_log_dict)

        return loss_dict, log_dict 

    def forward(
        self,
        generator_turn=False,
        guidance_turn=False,
        generator_data_dict=None, 
        guidance_data_dict=None
    ):    
        if generator_turn:
            loss_dict, log_dict = self.generator_forward(
                image=generator_data_dict["image"],
                text_embedding=generator_data_dict["text_embedding"],
                uncond_embedding=generator_data_dict["uncond_embedding"],
                unet_added_conditions=generator_data_dict["unet_added_conditions"],
                uncond_unet_added_conditions=generator_data_dict["uncond_unet_added_conditions"]
            )   
        elif guidance_turn:
            loss_dict, log_dict = self.guidance_forward(
                image=guidance_data_dict["image"],
                text_embedding=guidance_data_dict["text_embedding"],
                uncond_embedding=guidance_data_dict["uncond_embedding"],
                real_train_dict=guidance_data_dict["real_train_dict"],
                unet_added_conditions=guidance_data_dict["unet_added_conditions"],
                uncond_unet_added_conditions=guidance_data_dict["uncond_unet_added_conditions"]
            ) 
        else:
            raise NotImplementedError

        return loss_dict, log_dict 