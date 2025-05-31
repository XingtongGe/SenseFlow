# SenseFlow: Scaling Distribution Matching for Flow-based Text-to-Image Distillation

[Xingtong Ge](https://xingtongge.github.io/), Xin Zhang, [Tongda Xu](https://inkosizhong.github.io/), [Yi Zhang](https://zhangyi-3.github.io/), [Xinjie Zhang](https://xinjie-q.github.io/), [Yan Wang](https://yanwang202199.github.io/), [Jun Zhang](https://eejzhang.people.ust.hk/)

## Abstract

The Distribution Matching Distillation (DMD) has been successfully applied to text-to-image diffusion models such as Stable Diffusion (SD) 1.5. However, vanilla DMD suffers from convergence difficulties on large-scale flow-based text-to-image models, such as SD 3.5 and FLUX. In this paper, we first analyze the issues when applying vanilla DMD on large-scale models. Then, to overcome the scalability challenge, we propose implicit distribution alignment (IDA) to regularize the distance between the generator and fake distribution. Furthermore, we propose intra-segment guidance (ISG) to relocate the timestep importance distribution from the teacher model. With IDA alone, DMD converges for SD 3.5; employing both IDA and ISG, DMD converges for SD 3.5 and FLUX.1 dev. Along with other improvements such as scaled up discriminator models, our final model, dubbed **SenseFlow**, achieves superior performance in distillation for both diffusion based text-to-image models such as SDXL, and flow-matching models such as SD 3.5 Large and FLUX.

![](imgs/Fig1_final.png)



### 🌟Table 1: Quantitative Results on COCO-5K Dataset

**Bold** = best, <u>Underline</u> = second best. All results on 4-step generation.

#### Stable Diffusion XL Comparison

| Method           | NFE ↓ | FID-T ↓ | Patch FID-T ↓ | CLIP ↑ | HPSv2 ↑ | Pick ↑ | ImageReward ↑ |
|------------------|--------|----------|----------------|--------|---------|---------|----------------|
| SDXL             | 80     | --       | --             | 0.3293 | 0.2930  | 22.67   | 0.8719         |
| LCM-SDXL         | 4      | 18.47    | 30.63          | 0.3230 | 0.2824  | 22.22   | 0.5693         |
| PCM-SDXL         | 4      | 14.38    | 17.77          | 0.3242 | 0.2920  | 22.54   | 0.6926         |
| Flash-SDXL       | 4      | 17.97    | 23.24          | 0.3216 | 0.2830  | 22.17   | 0.4295         |
| SDXL-Lightning   | 4      | **13.67**| **16.57**      | 0.3214 | 0.2931  | 22.80   | 0.7799         |
| Hyper-SDXL       | 4      | <u>13.71</u>  | <u>17.49</u>        | 0.3254 | <u>0.3000</u> | <u>22.98</u> | <u>0.9777</u> |
| DMD2-SDXL        | 4      | 15.04    | 18.72          | **0.3277** | 0.2963 | <u>22.98</u> | 0.9324         |
| Ours-SDXL        | 4      | 17.76    | 21.01          | 0.3248 | **0.3010** | **23.17** | **0.9951** |

#### Stable Diffusion 3.5 Comparison

| Method               | NFE ↓ | FID-T ↓ | Patch FID-T ↓ | CLIP ↑ | HPSv2 ↑ | Pick ↑ | ImageReward ↑ |
|----------------------|--------|----------|----------------|--------|---------|---------|----------------|
| SD 3.5 Large         | 100    | --       | --             | 0.3310 | 0.2993  | 22.98   | 1.1629         |
| SD 3.5 Large Turbo   | 4      | <u>13.58</u>  | 22.88          | 0.3262 | 0.2909  | 22.89   | 1.0116         |
| Ours-SD 3.5          | 4      | **13.38**| **17.48**      | <u>0.3286</u> | **0.3016** | **23.01** | <u>1.1713</u> |
| Ours-SD 3.5 (Euler)  | 4      | 15.24    | <u>20.26</u>        | **0.3287** | <u>0.3008</u> | <u>22.90</u> | **1.2062** |

#### FLUX Comparison

| Method            | NFE ↓ | FID-T ↓ | Patch FID-T ↓ | CLIP ↑ | HPSv2 ↑ | Pick ↑ | ImageReward ↑ |
|-------------------|--------|----------|----------------|--------|---------|---------|----------------|
| FLUX.1 dev        | 50     | --       | --             | 0.3202 | 0.3000  | 23.18   | 1.1170         |
| FLUX.1 dev        | 25     | --       | --             | 0.3207 | 0.2986  | 23.14   | 1.1063         |
| FLUX.1-schnell    | 4      | --       | --             | **0.3264** | 0.2962 | 22.77   | 1.0755         |
| Hyper-FLUX        | 4      | <u>11.24</u>  | 23.47          | <u>0.3238</u> | 0.2963  | 23.09   | <u>1.0983</u> |
| FLUX-Turbo-Alpha  | 4      | **11.22**| 24.52          | 0.3218 | 0.2907  | 22.89   | 1.0106         |
| Ours-FLUX         | 4      | 15.64    | **19.60**      | 0.3167 | <u>0.2997</u> | <u>23.13</u> | 1.0921         |
| Ours-FLUX (Euler) | 4      | 16.50    | <u>20.29</u>        | 0.3171 | **0.3008** | **23.26** | **1.1424**     |

---



### 🌟Table 2: 4-Step Results on T2I-CompBench

**Bold**, <u>Underline</u> = best and second best (per teacher).

| Method                | Color | Shape | Texture | Spatial | Non-spatial | Complex-3-in-1 |
|------------------------|--------|--------|----------|----------|---------------|----------------|
| LCM-SDXL               | 0.5997 | 0.4015 | 0.4958   | 0.1672   | 0.3010        | 0.3364         |
| SDXL-Lightning         | 0.5758 | 0.4492 | 0.5154   | 0.2124   | 0.3098        | <u>0.3517</u>  |
| Hyper-SDXL            | <u>0.6435</u> | <u>0.4732</u> | <u>0.5581</u> | <u>0.2213</u> | **0.3104**    | 0.3301         |
| PCM-SDXL              | 0.5591 | 0.4142 | 0.4693   | 0.2013   | 0.3099        | 0.3234         |
| DMD2-SDXL             | 0.5811 | 0.4477 | 0.5175   | 0.2124   | 0.3098        | 0.3301         |
| **Ours-SDXL**         | **0.6867** | **0.4828** | **0.5989** | **0.2224** | <u>0.3100</u> | **0.3594**     |
| SD 3.5 Large Turbo    | 0.7050 | 0.5443 | 0.6512   | 0.2839   | 0.3130        | 0.3520         |
| Ours-SD 3.5           | <u>0.7657</u> | <u>0.6069</u> | <u>0.7427</u> | **0.2970** | <u>0.3177</u> | <u>0.3916</u> |
| **Ours-SD 3.5 (Euler)**| **0.7711** | **0.6149** | **0.7543** | <u>0.2857</u> | **0.3182** | **0.3968**     |
| FLUX.1-schnell        | 0.7317 | 0.5649 | 0.6919   | 0.2626   | 0.3122        | 0.3669         |
| Hyper-FLUX            | **0.7465** | 0.5023 | **0.6153** | **0.2945** | **0.3116** | **0.3766**     |
| FLUX-Turbo-Alpha      | <u>0.7406</u> | 0.4873 | 0.6024   | 0.2501   | <u>0.3094</u> | 0.3688         |
| Ours-FLUX             | 0.7284 | <u>0.5055</u> | 0.6031   | 0.2451   | 0.3028        | 0.3652         |
| Ours-FLUX (Euler)     | 0.7363 | **0.5120** | <u>0.6112</u> | <u>0.2521</u> | 0.3028        | <u>0.3697</u>  |
