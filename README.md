# SenseFlow: Scaling Distribution Matching for Flow-based Text-to-Image Distillation

[Xingtong Ge](https://xingtongge.github.io/), Xin Zhang, [Tongda Xu](https://inkosizhong.github.io/), [Yi Zhang](https://zhangyi-3.github.io/), [Xinjie Zhang](https://xinjie-q.github.io/), [Yan Wang](https://yanwang202199.github.io/), [Jun Zhang](https://eejzhang.people.ust.hk/)

## Abstract

The Distribution Matching Distillation (DMD) has been successfully applied to text-to-image diffusion models such as Stable Diffusion (SD) 1.5. However, vanilla DMD suffers from convergence difficulties on large-scale flow-based text-to-image models, such as SD 3.5 and FLUX. In this paper, we first analyze the issues when applying vanilla DMD on large-scale models. Then, to overcome the scalability challenge, we propose implicit distribution alignment (IDA) to regularize the distance between the generator and fake distribution. Furthermore, we propose intra-segment guidance (ISG) to relocate the timestep importance distribution from the teacher model. With IDA alone, DMD converges for SD 3.5; employing both IDA and ISG, DMD converges for SD 3.5 and FLUX.1 dev. Along with other improvements such as scaled up discriminator models, our final model, dubbed **SenseFlow**, achieves superior performance in distillation for both diffusion based text-to-image models such as SDXL, and flow-matching models such as SD 3.5 Large and FLUX.

![](imgs/Fig1_final.png)
