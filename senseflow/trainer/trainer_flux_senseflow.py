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
from senseflow.utils import instantiate_from_config
from senseflow.trainer.senseflow_utils import extract_into_tensor
from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    FluxTransformer2DModel,
)
from transformers import CLIPTokenizer, PretrainedConfig, T5TokenizerFast
__conditioning_keys__ = {"concat": "c_concat", "crossattn": "c_crossattn", "adm": "y"}
from PIL import Image
import torch.nn as nn
from senseflow.data.senseflow_dataset import SDImageDatasetLMDB, SDImageDatasetLMDBwoTokenizer, LaionText2ImageDataset
from senseflow.data.senseflow_dataset import cycle

# Copied from diffusers.examples.dreambooth.train_dreambooth_flux
def tokenize_prompt(tokenizer, prompt, max_sequence_length):
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        return_length=False,
        return_overflowing_tokens=False,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    return text_input_ids

def _encode_prompt_with_t5(
    text_encoder,
    tokenizer,
    max_sequence_length=512,
    prompt=None,
    num_images_per_prompt=1,
    device=None,
    text_input_ids=None,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    if tokenizer is not None:
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_length=False,
            return_overflowing_tokens=False,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
    else:
        if text_input_ids is None:
            raise ValueError("text_input_ids must be provided when the tokenizer is not specified")

    prompt_embeds = text_encoder(text_input_ids.to(device))[0]

    dtype = text_encoder.dtype
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

    _, seq_len, _ = prompt_embeds.shape

    # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

    return prompt_embeds

def _encode_prompt_with_clip(
    text_encoder,
    tokenizer,
    prompt: str,
    device=None,
    text_input_ids=None,
    num_images_per_prompt: int = 1,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    if tokenizer is not None:
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_overflowing_tokens=False,
            return_length=False,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids
    else:
        if text_input_ids is None:
            raise ValueError("text_input_ids must be provided when the tokenizer is not specified")

    prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=False)

    # Use pooled output of CLIPTextModel
    prompt_embeds = prompt_embeds.pooler_output
    prompt_embeds = prompt_embeds.to(dtype=text_encoder.dtype, device=device)

    # duplicate text embeddings for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, -1)

    return prompt_embeds

@torch.no_grad()
def encode_prompt(
    prompt: str,
    text_encoders,
    tokenizers,
    max_sequence_length=512,
    device=None,
    num_images_per_prompt: int = 1,
    text_input_ids_list=None,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)
    dtype = text_encoders[0].dtype
    device = device if device is not None else text_encoders[1].device
    pooled_prompt_embeds = _encode_prompt_with_clip(
        text_encoder=text_encoders[0],
        tokenizer=tokenizers[0],
        prompt=prompt,
        device=device,
        num_images_per_prompt=num_images_per_prompt,
        text_input_ids=text_input_ids_list[0] if text_input_ids_list else None,
    )

    prompt_embeds = _encode_prompt_with_t5(
        text_encoder=text_encoders[1],
        tokenizer=tokenizers[1],
        max_sequence_length=max_sequence_length,
        prompt=prompt,
        num_images_per_prompt=num_images_per_prompt,
        device=device,
        text_input_ids=text_input_ids_list[1] if text_input_ids_list else None,
    )

    text_ids = torch.zeros(batch_size, prompt_embeds.shape[1], 3).to(device=device, dtype=dtype)
    text_ids = text_ids.repeat(num_images_per_prompt, 1, 1)

    return {'prompt_embeds': prompt_embeds, 'pooled_prompt_embeds': pooled_prompt_embeds, 'text_ids': text_ids}
        
def import_model_class_from_model_name_or_path(
    pretrained_teacher_model: str, revision: str, subfolder: str = "text_encoder"
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_teacher_model, subfolder=subfolder, revision=revision
    )
    model_class = text_encoder_config.architectures[0]
    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "T5EncoderModel":
        from transformers import T5EncoderModel

        return T5EncoderModel
    else:
        raise ValueError(f"{model_class} is not supported.")

def predict_noise(dit, noisy_latents, text_embeddings, uncond_embedding, timesteps, 
    guidance_scale=1.0, pooled_prompt_embeds=None, uncond_pooled_prompt_embeds=None
):
    CFG_GUIDANCE = guidance_scale > 1

    if CFG_GUIDANCE:
        model_input = torch.cat([noisy_latents] * 2) 
        embeddings = torch.cat([uncond_embedding, text_embeddings]) 
        timesteps = torch.cat([timesteps] * 2) 
        pooled_embeds = torch.cat([uncond_pooled_prompt_embeds, pooled_prompt_embeds])

        noise_pred = dit(model_input, timestep=timesteps, encoder_hidden_states=embeddings, pooled_projections=pooled_embeds).sample
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond) 
    else:
        model_input = noisy_latents 
        embeddings = text_embeddings
        timesteps = timesteps
        pooled_embeds = pooled_prompt_embeds
        noise_pred = dit(model_input, timestep=timesteps, encoder_hidden_states=embeddings, pooled_projections=pooled_embeds).sample

    return noise_pred

# Copied from diffusers.pipelines.flux
def _prepare_latent_image_ids(batch_size, height, width, device, dtype):
    latent_image_ids = torch.zeros(height, width, 3)
    latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height)[:, None]
    latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width)[None, :]

    latent_image_id_height, latent_image_id_width, latent_image_id_channels = latent_image_ids.shape

    latent_image_ids = latent_image_ids.reshape(
        latent_image_id_height * latent_image_id_width, latent_image_id_channels
    )

    return latent_image_ids.to(device=device, dtype=dtype)

def _pack_latents(latents, batch_size, num_channels_latents, height, width):
    latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels_latents * 4)

    return latents

def _unpack_latents(latents, height, width, vae_scale_factor):
    batch_size, num_patches, channels = latents.shape

    # VAE applies 8x compression on images but we must also account for packing which requires
    # latent height and width to be divisible by 2.
    height = 2 * (int(height) // (vae_scale_factor * 2))
    width = 2 * (int(width) // (vae_scale_factor * 2))

    latents = latents.view(batch_size, height // 2, width // 2, channels // 4, 2, 2)
    latents = latents.permute(0, 3, 1, 4, 2, 5)

    latents = latents.reshape(batch_size, channels // (2 * 2), height, width)

    return latents

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
        self.build_dmd2_img_dataloader(global_batch_size = self.world_size)
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

    def build_dmd2_img_dataloader(
            self,
            global_batch_size: int = 8,
            num_workers: int = 32,
            pin_memory: bool = False,
            persistent_workers: bool = True,
            ):
        dataset_path = self.config["paths"]["dataset"]
        self.real_dataset = LaionText2ImageDataset(json_path=dataset_path, repeat=1)
        sampler = DistributedSampler(self.real_dataset, num_replicas=self.world_size, rank=self.local_rank, shuffle=True)
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
        self.backward_simulation = True
        self.denoising_timestep = 1000
        self.num_denoising_step = 4
        self.num_train_timesteps = 1000
        self.denoising_step_list = torch.tensor([1000.0, 904.0, 759.0, 512.0]).to('cuda')
        
        from argparse import Namespace
        args = Namespace()
        args.pretrained_teacher_model = self.config["paths"]["pretrained_model"]
        args.flux_wo_guidance_embed = self.config["paths"]["flux_wo_guidance_embed"]
        args.revision = None
        args.variant = None
        args.pretrained_vae_model_name_or_path = None
        args.pretrained_unet_lcm_path = None
        args.num_train_timesteps = 1000
        args.min_step_percent = 0.02
        args.max_step_percent = 0.98
        args.real_guidance_scale = 4.0
        args.fake_guidance_scale = 1.0
        args.diffusion_gan = True
        args.diffusion_gan_max_timestep = 1000
        # self.weight_dtype = torch.float32
        self.weight_dtype = torch.bfloat16

        self.sdxl_lora = False
        self.disable_sdxl_crossattn = False # True
        self.allin_bf16 = True
        self.laion_crop_size = 1024 # 768 # 512 # 256 # 768 # 1024 # 512
        print('laion crop size', self.laion_crop_size, 'all in bf16', self.allin_bf16)
        print('sdxl lora', self.sdxl_lora, 'crossatt disable', self.disable_sdxl_crossattn, 'lcm path', args.pretrained_unet_lcm_path)
        if args.pretrained_unet_lcm_path is not None:
            print('using lcm pretraining')
        

        # load FLUX vae
        vae_path = (
            args.pretrained_teacher_model
            if args.pretrained_vae_model_name_or_path is None
            else args.pretrained_vae_model_name_or_path
        )
        vae = AutoencoderKL.from_pretrained(
            vae_path,
            subfolder="vae" if args.pretrained_vae_model_name_or_path is None else None,
            revision=args.revision,
            torch_dtype=self.weight_dtype,
        )
        vae.enable_gradient_checkpointing()
        
        # load FLUX text encoders
        tokenizer_one = CLIPTokenizer.from_pretrained(
            args.pretrained_teacher_model,
            subfolder="tokenizer",
            revision=args.revision,
        )
        tokenizer_two = T5TokenizerFast.from_pretrained(
            args.pretrained_teacher_model,
            subfolder="tokenizer_2",
            revision=args.revision,
        )


        # 3. Load text encoders
        # import correct text encoder classes
        text_encoder_cls_one = import_model_class_from_model_name_or_path(
            args.pretrained_teacher_model, args.revision
        )
        text_encoder_cls_two = import_model_class_from_model_name_or_path(
            args.pretrained_teacher_model, args.revision, subfolder="text_encoder_2"
        )

        text_encoder_one = text_encoder_cls_one.from_pretrained(
            args.pretrained_teacher_model,
            subfolder="text_encoder",
            revision=args.revision,
            variant=args.variant,
        )
        text_encoder_two = text_encoder_cls_two.from_pretrained(
            args.pretrained_teacher_model,
            subfolder="text_encoder_2",
            revision=args.revision,
            variant=args.variant,
        )

        text_encoder_one.requires_grad_(False).eval().to(self.weight_dtype).cuda()
        text_encoder_two.requires_grad_(False).eval().to(self.weight_dtype).cuda()
        text_encoders = [text_encoder_one, text_encoder_two]
        tokenizers = [tokenizer_one, tokenizer_two]
        self.compute_embeddings_fn = functools.partial(
                                                    encode_prompt,
                                                    text_encoders=text_encoders,
                                                    tokenizers=tokenizers,
                                                )

        # load generator
        my_auto_wrap_policy = functools.partial(
            size_based_auto_wrap_policy, min_num_params=0
        )
        print('self.local_rank: ', self.local_rank)
        transformer = FluxTransformer2DModel.from_pretrained(
            args.flux_wo_guidance_embed, subfolder="transformer", revision=args.revision, variant=args.variant
        ).to(self.weight_dtype)
        transformer.enable_gradient_checkpointing()
        self.model = FSDP(
            transformer.cuda(),
            device_id=self.local_rank,
            sharding_strategy=torch.distributed.fsdp.ShardingStrategy.FULL_SHARD,
            auto_wrap_policy=my_auto_wrap_policy,
        )
        # guidance model
        fake_transformer = FluxTransformer2DModel.from_pretrained(
            args.flux_wo_guidance_embed, subfolder="transformer", revision=args.revision, variant=args.variant
        ).to(self.weight_dtype)
        fake_transformer.enable_gradient_checkpointing()
        fake_transformer.requires_grad_(True)
        real_transformer = FluxTransformer2DModel.from_pretrained(
            args.pretrained_teacher_model,
            subfolder="transformer",
            revision=args.revision,
            variant=args.variant,
        ).to(self.weight_dtype)
        real_transformer.enable_gradient_checkpointing()
        real_transformer.requires_grad_(False)

        self.guidance_model = GuidanceModel(fake_unet=fake_transformer, real_unet=real_transformer, args=args)
        self.guidance_model = FSDP(
            self.guidance_model.cuda(),
            device_id=self.local_rank,
            sharding_strategy=torch.distributed.fsdp.ShardingStrategy.FULL_SHARD,
            auto_wrap_policy=my_auto_wrap_policy,
        )

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


        # VFM GAN

        from senseflow.models.vfmgan import ProjectedDiscriminatorPlus, GANLoss
        from senseflow.models.clip import CLIP

        dino_name = 'vit_large_patch14_dinov2.lvd142m' 
        # hooks = [5, 11, 17, 23]
        hooks = [2, 5, 8, 10, 14, 19, 23]
        fix_res_dino = False
        crop_plan = 'none'
        p_crop, use_checkpoint, dis_useatt, ret_cls = 0.5, False, False, False
        dis_conv2d, dsample, diffaug, dino_pretrain = True, 3, True, True
        self.net_d = torch.nn.parallel.DistributedDataParallel(
            ProjectedDiscriminatorPlus(c_dim=768, dino_name=dino_name, hooks=hooks, crop_plan=crop_plan, p_crop=p_crop, use_checkpoint=use_checkpoint, 
                fix_res_dino=fix_res_dino, useatt=dis_useatt, ret_cls=ret_cls, conv2d=dis_conv2d, downsample=dsample, diffaug=diffaug, dino_pretrain=dino_pretrain).cuda(),
            device_ids=[self.local_rank],
        )
        self.net_d.train()
        self.cri_gan = GANLoss(
            gan_type="hinge",
            real_label_val=1.0,
            fake_label_val=0.0,
            loss_weight=1.0,
        )
        self.clip = CLIP().cuda().eval().requires_grad_(False)

        self.use_hybrid = False
        self.use_full_shard = True

        if self.use_hybrid and self.use_full_shard:
            raise NotImplementedError

        print("use hybrid: {}".format(self.use_hybrid))


        # noise scheduler
        self.noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            args.pretrained_teacher_model, subfolder="scheduler"
        )


    def build_optimizer(self):
        assert "optimizer" in self.config, "config does not contain key 'optimizer'"
        optimizer_config = self.config["optimizer"]

        use_8bit_adamw = True
        print('** 8bit adamw', use_8bit_adamw)
        if use_8bit_adamw:
            import bitsandbytes as bnb
            self.optimizer = bnb.optim.AdamW8bit(self.model.parameters(), lr=1.0, 
                betas=(optimizer_config["beta1"], optimizer_config["beta2"]),
                weight_decay=optimizer_config["weight_decay"], eps=optimizer_config["eps"],)
            self.optimizer_guidance = bnb.optim.AdamW8bit([param for param in self.guidance_model.parameters() if param.requires_grad], lr=1.0, 
                betas=(optimizer_config["beta1"], optimizer_config["beta2"]),
                weight_decay=optimizer_config["weight_decay"], eps=optimizer_config["eps"],)
            self.optimizer_d = bnb.optim.AdamW8bit(
                list(self.net_d.module.heads.parameters()),
                lr=1.0,
                betas=(optimizer_config["beta1"], optimizer_config["beta2"]),
                eps=optimizer_config["eps"],
                weight_decay=optimizer_config["weight_decay"],
            )
        else:
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
            self.optimizer_d = torch.optim.AdamW(
                list(self.net_d.module.heads.parameters()),
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
        d_lr_scheduler = instantiate_from_config(scheduler_config)
        self.d_lr_scheduler = LambdaLR(self.optimizer_d, lr_lambda=d_lr_scheduler.schedule)

    @torch.no_grad()
    def encode_first_stage_model(self, data):
        images = data.cuda()
        batch_size = images.shape[0]
        output = self.vae.encode(images).latent_dist.sample()
        output = (output - self.vae.config.shift_factor) * self.vae.config.scaling_factor
        return output.to(self.weight_dtype)
    
    

    @torch.no_grad()
    def sample_backward(self, noisy_image, denoising_cond_dict):
        batch_size = noisy_image.shape[0]
        device = noisy_image.device

        # we choose a random step and share it across all gpu
        selected_step = torch.randint(low=0, high=self.num_denoising_step, size=(1,), device=device, dtype=torch.long)

        generated_image = noisy_image

        vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        latent_image_ids = _prepare_latent_image_ids(
            noisy_image.shape[0],
            noisy_image.shape[2] // 2,
            noisy_image.shape[3] // 2,
            'cuda',
            self.weight_dtype,
        )

        for constant in self.denoising_step_list[:selected_step]:
            # current_sigmas = self.sigmas[constant]
            current_timesteps = constant
            current_sigmas = (current_timesteps / 1000.0).to(torch.bfloat16)
            cur_noise = torch.randn_like(
                generated_image.unsqueeze(dim=0).repeat(self.world_size, 1, 1, 1, 1)
            )[self.rank].cuda()

            # noisy_image = current_sigmas * torch.randn_like(generated_image) + (1.0 - current_sigmas) * generated_image
            noisy_image = current_sigmas * cur_noise + (1.0 - current_sigmas) * generated_image

            packed_noisy_model_input = _pack_latents(
                noisy_image,
                batch_size=noisy_image.shape[0],
                num_channels_latents=noisy_image.shape[1],
                height=noisy_image.shape[2],
                width=noisy_image.shape[3],
            )
            current_timesteps = torch.ones(batch_size, device=device, dtype=torch.long) * current_sigmas * self.noise_scheduler.config.num_train_timesteps

            model_pred = self.model(
                hidden_states=packed_noisy_model_input.to(self.weight_dtype),
                timestep=current_timesteps/1000.0,
                pooled_projections=denoising_cond_dict['pooled_prompt_embeds'].to(self.weight_dtype),
                encoder_hidden_states=denoising_cond_dict['prompt_embeds'].to(self.weight_dtype),
                txt_ids=denoising_cond_dict['text_ids'].to(self.weight_dtype),
                img_ids=latent_image_ids.to(self.weight_dtype),
                return_dict=False,
            )[0]

            model_pred = _unpack_latents(
                model_pred,
                height=noisy_image.shape[2] * vae_scale_factor,
                width=noisy_image.shape[3] * vae_scale_factor,
                vae_scale_factor=vae_scale_factor,
            )

            generated_image = (noisy_image - current_sigmas * model_pred).to(noisy_image.dtype)

        return_timesteps = self.denoising_step_list[selected_step] * torch.ones(batch_size, device=device, dtype=torch.long)
        return generated_image, return_timesteps, selected_step

    @torch.no_grad()
    def prepare_denoising_data(self, denoising_dict, real_train_dict, noise):

        indices = torch.randint(
            0, self.num_denoising_step, (noise.shape[0],), device=noise.device, dtype=torch.long
        )
        timesteps = self.denoising_step_list.to(noise.device)[indices]
        denoising_cond_dict = self.compute_embeddings_fn(denoising_dict['caption'] * noise.shape[0])

        if real_train_dict is not None:
            real_cond_dict = self.compute_embeddings_fn(real_train_dict['caption'] * noise.shape[0])
            real_train_dict['text_embedding'] = real_cond_dict['prompt_embeds']
            real_train_dict['pooled_prompt_embeds'] = real_cond_dict['pooled_prompt_embeds']
            real_train_dict['text_ids'] = real_cond_dict['text_ids']

        if self.backward_simulation:
            cur_noise = torch.randn_like(
                noise.unsqueeze(dim=0).repeat(self.world_size, 1, 1, 1, 1)
            )[self.rank].cuda()
            clean_images, timesteps, indices = self.sample_backward(cur_noise, denoising_cond_dict)

        else:
            clean_images = denoising_dict['image'].to(noise.device)
        # print('cur_sigmas: ', cur_sigmas)
        cur_sigmas = (timesteps / 1000.0).to(torch.bfloat16)

        noisy_image = cur_sigmas * noise + (1.0 - cur_sigmas) * clean_images

        return timesteps, cur_sigmas, denoising_cond_dict, real_train_dict, noisy_image, clean_images, indices

    def differentiable_decode_first_stage(self, z):
        z = 1.0 / self.scaling_factor * z + self.vae.config.shift_factor
        return self.vae.decode(z).sample

    def append_dims(self, x, target_dims):
        """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
        dims_to_append = target_dims - x.ndim
        if dims_to_append < 0:
            raise ValueError(f"input has {x.ndim} dims but target_dims is {target_dims}, which is less")
        return x[(...,) + (None,) * dims_to_append]
    
    def get_isg_guidance(
            self,
            noisy_image,
            packed_noisy_model_input,
            latent_image_ids,
            h,
            w,
            timestep_indice,
            generator_pred,
            cond_dict,
        ):
        """
        Intra-Segment Guidance (ISG) loss computation for FLUX model.
        Samples an intermediate timestep between timestep and timestep_prev,
        and computes guidance loss by comparing generator's direct path with 
        the reference path through the intermediate timestep.
        """
        # Sample an intermediate timestep_mid between timestep and timestep_prev
        timestep = self.denoising_step_list[timestep_indice]

        if timestep_indice == 3:
            timestep_prev = torch.zeros_like(timestep).to(torch.float32).to('cuda')
        else:
            timestep_prev = self.denoising_step_list[timestep_indice+1]
        timestep_mid = torch.randint(torch.ceil(timestep_prev+20).int(), torch.floor(timestep-20).int()+1, (1,)).to(torch.float32).to('cuda')

        # Use guidance_model.real_unet to denoise from timestep to timestep_mid (no grad)
        with torch.no_grad():
            c = random.uniform(1, 8)
            guidance = torch.tensor([c], device='cuda')
            guidance = guidance.expand(noisy_image.shape[0])
            real_score_pred = self.guidance_model.real_unet(
                hidden_states=packed_noisy_model_input.to(self.weight_dtype),
                timestep=(timestep/1000.0).to(self.weight_dtype),
                guidance=guidance.to(self.weight_dtype),
                pooled_projections=cond_dict['pooled_prompt_embeds'].to(self.weight_dtype),
                encoder_hidden_states=cond_dict['prompt_embeds'].to(self.weight_dtype),
                txt_ids=cond_dict['text_ids'].to(self.weight_dtype),
                img_ids=latent_image_ids.to(self.weight_dtype),
                return_dict=False,
            )[0]
            real_score_pred = _unpack_latents(
                real_score_pred,
                height=int(h * self.vae_scale_factor),
                width=int(w * self.vae_scale_factor),
                vae_scale_factor=self.vae_scale_factor,
            )
            x_mid = noisy_image + ((timestep_mid - timestep)/1000.0 * real_score_pred).to(noisy_image.dtype)
            packed_x_mid = _pack_latents(
                x_mid,
                batch_size=x_mid.shape[0],
                num_channels_latents=x_mid.shape[1],
                height=x_mid.shape[2],
                width=x_mid.shape[3],
            )
            model_pred = self.model(
                hidden_states=packed_x_mid.to(self.weight_dtype),
                timestep=(timestep_mid/1000.0).to(self.weight_dtype),
                pooled_projections=cond_dict['pooled_prompt_embeds'].to(self.weight_dtype),
                encoder_hidden_states=cond_dict['prompt_embeds'].to(self.weight_dtype),
                txt_ids=cond_dict['text_ids'].to(self.weight_dtype),
                img_ids=latent_image_ids.to(self.weight_dtype),
                return_dict=False,
            )[0]
            model_pred = _unpack_latents(
                model_pred,
                height=int(h * self.vae_scale_factor),
                width=int(w * self.vae_scale_factor),
                vae_scale_factor=self.vae_scale_factor,
            )
            target_x_prev = x_mid + ((timestep_prev - timestep_mid)/1000.0 * model_pred).to(x_mid.dtype)
        generated_x_prev = noisy_image + ((timestep_prev - timestep)/1000.0 * generator_pred).to(noisy_image.dtype)
        isg_guidance_loss = torch.mean(torch.sqrt((generated_x_prev.float() - target_x_prev.float()) ** 2 + 0.001**2) - 0.001)
        return isg_guidance_loss, target_x_prev, generated_x_prev

    def train(self):
        # scaler = ShardedGradScaler()
        # guidance_scaler = ShardedGradScaler()
        # scaler_d = ShardedGradScaler()
        iter_time_list = []
        torch.cuda.synchronize()
        dist.barrier()
        iter_begin_time = time.time()

        self.num_total_iters = 10000000
        self.GD = 5  # TTUR
        self.latent_channel = 16
        self.latent_resolution = 128
        self.network_context_manager = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.gan_weight = 2.0
        self.ida_w = 0.97
        self.rank, self.world_size = dist.get_rank(), dist.get_world_size()
        current_iter = 0
        for index in range(self.num_total_iters):
            current_iter += 1                
            COMPUTE_GENERATOR_GRADIENT = current_iter % self.GD == 0
            self.backward_simulation = torch.rand(1).item() > 0.5  # 50% probability
            denoising_dict = next(self.denoising_dataloader)
            ori_image = denoising_dict['image'].cuda()
            denoising_dict['image'] = self.encode_first_stage_model(denoising_dict['image'].to(torch.bfloat16))
            
            latent_image_ids = _prepare_latent_image_ids(
                    denoising_dict['image'].shape[0],
                    denoising_dict['image'].shape[2] // 2,
                    denoising_dict['image'].shape[3] // 2,
                    'cuda',
                    self.weight_dtype,
                )
            noise = torch.randn(self.batch_size, self.latent_channel, self.latent_resolution, self.latent_resolution).cuda()
            noise = torch.randn_like(
                noise.unsqueeze(dim=0).repeat(self.world_size, 1, 1, 1, 1)
            )[self.rank].cuda()

            # generator forward
            timesteps, sigmas, denoising_cond_dict, _, noisy_image, clean_image, timestep_indice = self.prepare_denoising_data(
                    denoising_dict, None, noise
                )
            # current_timesteps = torch.ones(self.batch_size, device=noise.device, dtype=torch.long) * sigmas * self.noise_scheduler.config.num_train_timesteps
            # Build unconditional embeddings
            # uncond_prompt_embeds, uncond_pooled_prompt_embeds, uncond_text_ids = self.compute_embeddings_fn([''] * noise.shape[0])
            uncond_dict = self.compute_embeddings_fn([''] * noise.shape[0])

            packed_noisy_model_input = _pack_latents(noisy_image, batch_size=noisy_image.shape[0], num_channels_latents=noisy_image.shape[1], height=noisy_image.shape[2], width=noisy_image.shape[3])
                
            if COMPUTE_GENERATOR_GRADIENT:
                with self.network_context_manager:
                    model_pred = self.model(
                        hidden_states=packed_noisy_model_input,
                        timestep=timesteps / 1000.0,
                        pooled_projections=denoising_cond_dict['pooled_prompt_embeds'],
                        encoder_hidden_states=denoising_cond_dict['prompt_embeds'],
                        txt_ids=denoising_cond_dict['text_ids'],
                        img_ids=latent_image_ids,
                        return_dict=False,
                    )[0]
                    # upscaling height & width as discussed in https://github.com/huggingface/diffusers/pull/9257#discussion_r1731108042
                    model_pred = _unpack_latents(
                        model_pred,
                        height=int(noisy_image.shape[2] * self.vae_scale_factor),
                        width=int(noisy_image.shape[3] * self.vae_scale_factor),
                        vae_scale_factor=self.vae_scale_factor,
                    )

                    # ISG guidance
                    isg_guidance_loss, target_x_prev, generated_x_prev = self.get_isg_guidance(
                        noisy_image=noisy_image,
                        packed_noisy_model_input=packed_noisy_model_input,
                        latent_image_ids=latent_image_ids,
                        h = noisy_image.shape[2],
                        w = noisy_image.shape[3],
                        timestep_indice=timestep_indice,
                        generator_pred=model_pred,
                        cond_dict = denoising_cond_dict,
                    )
                    print('isg_guidance_loss: ', isg_guidance_loss)
            else:
                with torch.no_grad():
                    with self.network_context_manager:
                        # Predict the noise residual
                        model_pred = self.model(
                            hidden_states=packed_noisy_model_input,
                            timestep=timesteps / 1000.0,
                            pooled_projections=denoising_cond_dict['pooled_prompt_embeds'],
                            encoder_hidden_states=denoising_cond_dict['prompt_embeds'],
                            txt_ids=denoising_cond_dict['text_ids'],
                            img_ids=latent_image_ids,
                            return_dict=False,
                        )[0]
                        # upscaling height & width as discussed in https://github.com/huggingface/diffusers/pull/9257#discussion_r1731108042
                        model_pred = _unpack_latents(
                            model_pred,
                            height=int(noisy_image.shape[2] * self.vae_scale_factor),
                            width=int(noisy_image.shape[3] * self.vae_scale_factor),
                            vae_scale_factor=self.vae_scale_factor,
                        )

            generated_image = (noisy_image - sigmas * model_pred).to(noisy_image.dtype)

            with torch.no_grad():
                clipcond = self.clip.encode_text(denoising_dict["caption"])

            if COMPUTE_GENERATOR_GRADIENT:
                generator_data_dict = {
                    "image": generated_image,
                    "cond": denoising_cond_dict,
                    "uncond": uncond_dict,
                    "real_train_dict": None,
                }

                # avoid any side effects of gradient accumulation
                self.guidance_model.requires_grad_(False)
                generator_loss_dict, generator_log_dict = self.guidance_model(
                    generator_turn=True,
                    guidance_turn=False,
                    generator_data_dict=generator_data_dict
                )
                # Compute GAN loss for generator
                with torch.cuda.amp.autocast(enabled=True, dtype=torch.float32):
                    imgres = self.differentiable_decode_first_stage(generated_image.float())
                with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
                    fake_g_pred = self.net_d(imgres.float(), clipcond, ori_image.float())
            else:
                generator_loss_dict = {}
                generator_log_dict = {}
                with torch.no_grad():
                    with torch.cuda.amp.autocast(enabled=True, dtype=torch.float32):
                        imgres = self.differentiable_decode_first_stage(generated_image.float())
                    with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
                        fake_g_pred = self.net_d(imgres.float(), clipcond, ori_image.float())

            l_g_gan = self.gan_weight * self.cri_gan(
                        fake_g_pred, True, is_disc=False, keepdim=True
                    )
            timew = (1 - sigmas) ** 2
            # print('timew, timesteps:', timew, timesteps)
            generator_loss_dict['l_g_gan'] = (l_g_gan * timew).mean()

            generator_log_dict["guidance_data_dict"] = {
                "image": generated_image.detach(),
                "cond": denoising_cond_dict,
                "uncond": uncond_dict,
                # "real_train_dict": real_train_dict,
            }
            generator_log_dict['denoising_timestep'] = timesteps
            generator_log_dict['denoising_sigmas'] = sigmas

            generator_loss = 0.0 
            if COMPUTE_GENERATOR_GRADIENT:
                generator_loss += generator_loss_dict["loss_dm"]
                generator_loss += (timew * l_g_gan).mean()
                generator_loss += isg_guidance_loss
                
                generator_loss.backward()
                self.model.clip_grad_norm_(1.0)
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.optimizer_guidance.zero_grad()

                if self.ida_w != 1.0:
                    with torch.no_grad():
                        target_model_named_parm_dict = dict(self.guidance_model.fake_unet.named_parameters())
                        for name_A, param_A in self.model.named_parameters():
                            param_B = target_model_named_parm_dict[name_A]
                            if param_A.requires_grad:
                                param_B.data.mul_(self.ida_w).add_(param_A.data, alpha=1 - self.ida_w)

            self.lr_scheduler.step()

            self.guidance_model.requires_grad_(True)
            self.guidance_model.real_unet.requires_grad_(False)
            # update guidance model (dfake and classifier)
            guidance_loss_dict, guidance_log_dict = self.guidance_model(
                generator_turn=False,
                guidance_turn=True,
                guidance_data_dict=generator_log_dict['guidance_data_dict']
            )
            guidance_loss = 0 

            guidance_loss += guidance_loss_dict["loss_fake_mean"]

            guidance_loss.backward()
            self.guidance_model.clip_grad_norm_(1.0)
            self.optimizer_guidance.step()
            self.optimizer.zero_grad()
            self.optimizer_guidance.zero_grad()
            self.guidance_lr_scheduler.step()


            # train discriminator
            d_loss_dict = {}
            gt = ori_image
            with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
                real_d_pred, _ref, _fea = self.net_d(
                    gt, clipcond, ori_image, return_ref=True
                )
                l_d_real = self.cri_gan(real_d_pred, True, is_disc=True, keepdim=True)
                l_d_real = torch.mean(l_d_real * timew)
                d_loss_dict["l_d_real"] = l_d_real
                d_loss_dict["out_d_real"] = torch.mean(real_d_pred.detach())
                l_d_real.backward()

                # scaler_d.scale(l_d_real).backward()
                fake_d_pred = self.net_d(imgres.detach(), clipcond, ori_image)
                l_d_fake = self.cri_gan(fake_d_pred, False, is_disc=True, keepdim=True)
                l_d_fake = torch.mean(l_d_fake * timew)
                d_loss_dict["l_d_fake"] = l_d_fake
                d_loss_dict["out_d_fake"] = torch.mean(fake_d_pred.detach())

                l_d_fake.backward()

                torch.nn.utils.clip_grad_norm_(
                    self.net_d.module.heads.parameters(), 1.0
                )
                self.optimizer_d.step()
                self.optimizer_d.zero_grad()
                self.d_lr_scheduler.step()


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
            # print('model_pred: ', model_pred.shape)
            if current_iter % 15 == 0 and self.rank == 0:
                with torch.no_grad():
                    
                    batchid = 0
                    # print('generated_image: ', generated_image.shape, generated_image.max(), generated_image.min(), generated_image.device)
                    # print('clean_image: ', clean_image.shape, clean_image.max(), clean_image.min(), clean_image.device)
                    # print('denoising_dict: ', denoising_dict['image'].shape, denoising_dict['image'].max(), denoising_dict['image'].min(), denoising_dict['image'].device)
                    # # print('generator_log_dict: ', generator_log_dict['dmtrain_pred_real_image'].shape)
                    # print('guidance_log_dict: ', guidance_log_dict['faketrain_x0_pred'].shape, guidance_log_dict['faketrain_x0_pred'].max(), guidance_log_dict['faketrain_x0_pred'].min(), guidance_log_dict['faketrain_x0_pred'].device)

                    model_pred_img = self.differentiable_decode_first_stage(generated_image.to(self.weight_dtype))
                    clean_img = self.differentiable_decode_first_stage(clean_image.to(self.weight_dtype))
                    x_start_img = self.differentiable_decode_first_stage(denoising_dict['image'].to(self.weight_dtype).cuda())
                    if COMPUTE_GENERATOR_GRADIENT:
                        dmtrain_pred_real_image = self.differentiable_decode_first_stage(generator_log_dict['dmtrain_pred_real_image'].to(self.weight_dtype).cuda())
                        dmtrain_pred_fake_image = self.differentiable_decode_first_stage(generator_log_dict['dmtrain_pred_fake_image'].to(self.weight_dtype).cuda())
                        target_x_prev_img = self.differentiable_decode_first_stage(target_x_prev.to(self.weight_dtype).cuda()) #target_x_prev, generated_x_prev
                        generated_x_prev_img = self.differentiable_decode_first_stage(generated_x_prev.to(self.weight_dtype).cuda())
                    faketrain_x0_pred =  self.differentiable_decode_first_stage(guidance_log_dict['faketrain_x0_pred'].to(self.weight_dtype))
                    print(f"save image {denoising_dict['caption'][batchid]}")
                    if COMPUTE_GENERATOR_GRADIENT:
                        save_model_pred_img , save_clean_img, save_x_start_img, save_dmtrain_pred_real_image, save_dmtrain_pred_fake_image, save_faketrain_x0_pred, save_target_x_prev_img, save_generated_x_prev_img = \
                            model_pred_img[[batchid]].float(), clean_img[[batchid]].float(), x_start_img[[batchid]].float(), dmtrain_pred_real_image[[batchid]].float(), dmtrain_pred_fake_image[[batchid]].float(), faketrain_x0_pred[[batchid]].float(), target_x_prev_img[[batchid]].float(), generated_x_prev_img[[batchid]].float()
                    else:
                        save_model_pred_img , save_clean_img, save_x_start_img, save_faketrain_x0_pred = \
                            model_pred_img[[batchid]].float(), clean_img[[batchid]].float(), x_start_img[[batchid]].float(), faketrain_x0_pred[[batchid]].float()
                    th2np = lambda x: x[batchid].detach().permute(1, 2, 0).cpu().numpy()
                    np2save = lambda x: ((x + 1).clip(0, 2) / 2. * 255).astype(np.uint8)
                    res = []
                    if COMPUTE_GENERATOR_GRADIENT:
                        for item in [save_model_pred_img, save_clean_img, save_x_start_img, save_dmtrain_pred_real_image, save_dmtrain_pred_fake_image, save_faketrain_x0_pred, save_target_x_prev_img, save_generated_x_prev_img]:
                            res.append(np2save(th2np(item)))
                    else:
                        for item in [save_model_pred_img, save_clean_img, save_x_start_img, save_faketrain_x0_pred]:
                            res.append(np2save(th2np(item)))
                    os.makedirs(os.path.join(self.image_logger_save_dir, 'image_log'), exist_ok=True)
                    tpath = os.path.join(self.image_logger_save_dir, 'image_log', 'image_log-%d-%d.jpg' % (current_iter, generator_log_dict['denoising_timestep'][0].cpu().item()))
                    print('log img save', tpath)
                    res = np.concatenate(res, axis=1)
                    Image.fromarray(res).save(tpath)

            # if current_iter % self.log_interval == 0 and self.rank == 0:
            if current_iter % 15 == 0 and self.rank == 0:
                if COMPUTE_GENERATOR_GRADIENT:
                    print(
                        "iter {}/{}, generator_loss: {}, loss_dm: {}, isg_guidance_loss: {}, guidance_loss: {}, loss_fake_mean: {}, l_g_gan: {}, l_d_real: {}, l_d_fake: {}, lr: {}, guidance_lr: {}, iter time avg: {}, iter time: {}".format(
                            current_iter,
                            self.num_total_iters,
                            generator_loss,
                            generator_loss_dict["loss_dm"].mean().detach().cpu().item(),
                            # generator_log_dict['dmtrain_grad'],
                            isg_guidance_loss,
                            guidance_loss,
                            guidance_loss_dict["loss_fake_mean"],
                            float(generator_loss_dict["l_g_gan"].detach().cpu()),
                            float(d_loss_dict["l_d_real"].detach().cpu()),
                            float(d_loss_dict["l_d_fake"].detach().cpu()),
                            self.optimizer.param_groups[0]["lr"],
                            self.optimizer_guidance.param_groups[0]["lr"],
                            sum(iter_time_list) / len(iter_time_list),
                            iter_time,
                        )
                    )
                else:
                    print(
                        "iter {}/{}, generator_loss: {}, guidance_loss: {}, loss_fake_mean: {}, l_g_gan: {}, l_d_real: {}, l_d_fake: {}, lr: {}, guidance_lr: {}, iter time avg: {}, iter time: {}".format(
                            current_iter,
                            self.num_total_iters,
                            generator_loss,
                            guidance_loss,
                            guidance_loss_dict["loss_fake_mean"],
                            float(generator_loss_dict["l_g_gan"].detach().cpu()),
                            float(d_loss_dict["l_d_real"].detach().cpu()),
                            float(d_loss_dict["l_d_fake"].detach().cpu()),
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

from diffusers.training_utils import compute_density_for_timestep_sampling, compute_loss_weighting_for_sd3

class GuidanceModel(nn.Module):
    def __init__(self, fake_unet, real_unet, args):
        super().__init__()
        self.fake_unet = fake_unet
        self.real_unet = real_unet
        self.real_unet = self.real_unet.to(torch.bfloat16)
        # noise scheduler
        self.noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            args.pretrained_teacher_model, subfolder="scheduler"
        )
        # print('self.noise_scheduler.sigmas: ', self.noise_scheduler.sigmas)
        self.sigmas = torch.flip(self.noise_scheduler.sigmas, dims=[0]).to('cuda')

        self.num_train_timesteps = args.num_train_timesteps
        self.weight_dtype = torch.bfloat16
        self.vae_scale_factor = 8
        self.network_context_manager = torch.autocast(device_type="cuda", dtype=torch.bfloat16)


    def get_sigmas(self, timesteps, n_dim=4, dtype=torch.float32):
        sigmas = self.noise_scheduler.sigmas.to(device='cuda', dtype=dtype)
        schedule_timesteps = self.noise_scheduler.timesteps.cuda()
        timesteps = timesteps.cuda()
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma
    
    def compute_distribution_matching_loss(
        self,
        latents,
        cond,
        uncond = None,
    ):
        original_latents = latents 
        batch_size = latents.shape[0]
        with torch.no_grad():
            u = compute_density_for_timestep_sampling(
                    weighting_scheme='logit_normal',
                    batch_size=batch_size,
                    logit_mean=0.0,
                    logit_std=1.0,
                    mode_scale=1.29,
                )
            indices = (u * self.noise_scheduler.config.num_train_timesteps).long()
            timesteps = self.noise_scheduler.timesteps[indices].to(device=latents.device)
            # Add noise according to flow matching.
            # zt = (1 - texp) * x + texp * z1
            sigmas = self.get_sigmas(timesteps, n_dim=latents.ndim, dtype=latents.dtype)

            noise = torch.randn_like(latents)
            noisy_image = (1.0 - sigmas) * latents + sigmas * noise

            packed_noisy_model_input = _pack_latents(
                noisy_image,
                batch_size=noisy_image.shape[0],
                num_channels_latents=noisy_image.shape[1],
                height=noisy_image.shape[2],
                width=noisy_image.shape[3],
            )

            latent_image_ids = _prepare_latent_image_ids(
                    noisy_image.shape[0],
                    noisy_image.shape[2] // 2,
                    noisy_image.shape[3] // 2,
                    'cuda',
                    self.weight_dtype,
                )

            pred_fake_noise = self.fake_unet(
                hidden_states=packed_noisy_model_input.to(self.weight_dtype),
                timestep=timesteps/1000.0,
                pooled_projections=cond['pooled_prompt_embeds'].to(self.weight_dtype),
                encoder_hidden_states=cond['prompt_embeds'].to(self.weight_dtype),
                txt_ids=cond['text_ids'].to(self.weight_dtype),
                img_ids=latent_image_ids.to(self.weight_dtype),
                return_dict=False,
            )[0]

            pred_fake_noise = _unpack_latents(
                pred_fake_noise,
                height=noisy_image.shape[2] * self.vae_scale_factor,
                width=noisy_image.shape[3] * self.vae_scale_factor,
                vae_scale_factor=self.vae_scale_factor,
            )

            pred_fake_image = (noisy_image - sigmas * pred_fake_noise).to(noisy_image.dtype)
            c = random.uniform(1, 8)
            guidance = torch.tensor([c], device='cuda')
            guidance = guidance.expand(noisy_image.shape[0])

            pred_real_noise = self.real_unet(
                hidden_states=packed_noisy_model_input.to(self.weight_dtype),
                timestep=timesteps/1000.0,
                guidance=guidance,
                pooled_projections=cond['pooled_prompt_embeds'].to(self.weight_dtype),
                encoder_hidden_states=cond['prompt_embeds'].to(self.weight_dtype),
                txt_ids=cond['text_ids'].to(self.weight_dtype),
                img_ids=latent_image_ids.to(self.weight_dtype),
                return_dict=False,
            )[0]

            pred_real_noise = _unpack_latents(
                pred_real_noise,
                height=noisy_image.shape[2] * self.vae_scale_factor,
                width=noisy_image.shape[3] * self.vae_scale_factor,
                vae_scale_factor=self.vae_scale_factor,
            )

            pred_real_image = (noisy_image - sigmas * pred_real_noise).to(noisy_image.dtype)

            p_real = (latents - pred_real_image)
            p_fake = (latents - pred_fake_image)

            grad = (p_real - p_fake) / torch.abs(p_real).mean(dim=[1, 2, 3], keepdim=True) 
            grad = torch.nan_to_num(grad)

        loss = 0.5 * F.mse_loss(original_latents.float(), (original_latents-grad).detach().float(), reduction="mean")         

        loss_dict = {
            "loss_dm": loss 
        }

        dm_log_dict = {
            "dmtrain_noisy_latents": noisy_image.detach().float(),
            "dmtrain_pred_real_image": pred_real_image.detach().float(),
            "dmtrain_pred_fake_image": pred_fake_image.detach().float(),
            "dmtrain_grad": grad.detach().float(),
            "dmtrain_gradient_norm": torch.norm(grad).item()
        }

        return loss_dict, dm_log_dict
    
    def compute_loss_fake(
        self,
        latents,
        cond,
        uncond = None,
    ):
        latents = latents.detach()
        batch_size = latents.shape[0]
        u = compute_density_for_timestep_sampling(
                weighting_scheme='logit_normal',
                batch_size=batch_size,
                logit_mean=0.0,
                logit_std=1.0,
                mode_scale=1.29,
            )
        indices = (u * self.noise_scheduler.config.num_train_timesteps).long()
        timesteps = self.noise_scheduler.timesteps[indices].to(device=latents.device)
        # Add noise according to flow matching.
        # zt = (1 - texp) * x + texp * z1
        sigmas = self.get_sigmas(timesteps, n_dim=latents.ndim, dtype=latents.dtype)

        noise = torch.randn_like(latents)
        noisy_image = (1.0 - sigmas) * latents + sigmas * noise
        weighting = compute_loss_weighting_for_sd3(weighting_scheme='logit_normal', sigmas=sigmas)

        packed_noisy_model_input = _pack_latents(
                noisy_image,
                batch_size=noisy_image.shape[0],
                num_channels_latents=noisy_image.shape[1],
                height=noisy_image.shape[2],
                width=noisy_image.shape[3],
            )

        latent_image_ids = _prepare_latent_image_ids(
                noisy_image.shape[0],
                noisy_image.shape[2] // 2,
                noisy_image.shape[3] // 2,
                'cuda',
                self.weight_dtype,
            )

        pred_fake_noise = self.fake_unet(
            hidden_states=packed_noisy_model_input.to(self.weight_dtype),
            timestep=timesteps/1000.0,
            pooled_projections=cond['pooled_prompt_embeds'].to(self.weight_dtype),
            encoder_hidden_states=cond['prompt_embeds'].to(self.weight_dtype),
            txt_ids=cond['text_ids'].to(self.weight_dtype),
            img_ids=latent_image_ids.to(self.weight_dtype),
            return_dict=False,
        )[0]

        pred_fake_noise = _unpack_latents(
            pred_fake_noise,
            height=noisy_image.shape[2] * self.vae_scale_factor,
            width=noisy_image.shape[3] * self.vae_scale_factor,
            vae_scale_factor=self.vae_scale_factor,
        )

        pred_fake_noise = pred_fake_noise.float()
        target = noise - latents
        loss_fake = torch.mean(
            (weighting.float() * (pred_fake_noise - target.float()) ** 2).reshape(target.shape[0], -1),
            1,
        )
        loss_fake = loss_fake.mean()

        fake_x0_pred = (noisy_image - sigmas * pred_fake_noise).to(noisy_image.dtype)

        loss_dict = {
            "loss_fake_mean": loss_fake,
        }

        fake_log_dict = {
            "faketrain_latents": latents.detach().float(),
            "faketrain_noisy_latents": noisy_image.detach().float(),
            "faketrain_x0_pred": fake_x0_pred.detach().float()
        }
        return loss_dict, fake_log_dict

    def generator_forward(
        self,
        image,
        cond,
        uncond,
    ):
        loss_dict = {}
        log_dict = {}

        # image.requires_grad_(True)
        dm_dict, dm_log_dict = self.compute_distribution_matching_loss(
            image, cond, uncond,
        )

        loss_dict.update(dm_dict)
        log_dict.update(dm_log_dict)

        return loss_dict, log_dict 
    
    def guidance_forward(
        self,
        image,
        cond,
        uncond,
        # real_train_dict=None,
    ):
        fake_dict, fake_log_dict = self.compute_loss_fake(
            image, cond, uncond
        )

        loss_dict = fake_dict 
        log_dict = fake_log_dict

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
                cond=generator_data_dict["cond"],
                uncond=generator_data_dict["uncond"]
            )
        elif guidance_turn:
            loss_dict, log_dict = self.guidance_forward(
                image=guidance_data_dict["image"],
                cond=guidance_data_dict["cond"],
                uncond=guidance_data_dict["uncond"],
            )
        else:
            raise NotImplementedError

        return loss_dict, log_dict 
    
