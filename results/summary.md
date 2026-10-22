# Experiment summary

Setup: `stabilityai/stable-diffusion-xl-base-1.0`, 1024×1024, 30 steps, CFG 5.0, scheduler euler, fp16, device NVIDIA A40, torch 2.14.1+cu126, diffusers 0.39.0

## Main comparison (`main`, n=150 prompts)

| method | n | CLIPScore | 95% CI | Δ vs base | Wilcoxon p | time (s/img) | rel. time | peak mem (GB) |
|---|---|---|---|---|---|---|---|---|
| SDXL | 150 | 26.56 | [26.07, 27.06] | — | — | 6.40 ± 0.02 | 1.00× | 10.49 |
| SDXL+FreeU | 150 | 26.40 | [25.82, 26.96] | -0.16 | 0.333 | 6.58 ± 0.04 | 1.03× | 10.49 |
| SDXL+PAG | 150 | 26.00 | [25.48, 26.50] | -0.56 | 0.00206 | 9.41 ± 0.01 | 1.47× | 10.49 |
| FreeU+PAG | 150 | 25.96 | [25.41, 26.53] | -0.60 | 0.00547 | 9.60 ± 0.03 | 1.50× | 10.49 |

### CLIPScore by category

| category | SDXL | SDXL+FreeU | SDXL+PAG | FreeU+PAG |
|---|---|---|---|---|
| animal | 27.54 | 27.54 | 27.43 | 27.38 |
| indoor | 26.26 | 25.37 | 25.55 | 25.29 |
| multi_object | 27.32 | 27.28 | 26.52 | 26.73 |
| outdoor | 25.35 | 25.0 | 24.34 | 24.36 |
| person | 26.32 | 26.78 | 26.17 | 26.03 |

## FreeU hyper-parameters (`freeu_sweep`, n=50 prompts)

| method | n | CLIPScore | 95% CI | Δ vs base | Wilcoxon p | time (s/img) | rel. time | peak mem (GB) |
|---|---|---|---|---|---|---|---|---|
| SDXL | 50 | 26.57 | [25.72, 27.40] | — | — | 6.39 ± 0.03 | 1.00× | 10.49 |
| FreeU (weak) | 50 | 26.68 | [25.76, 27.55] | +0.11 | 0.66 | 6.57 ± 0.02 | 1.03× | 10.49 |
| SDXL+FreeU | 50 | 26.61 | [25.56, 27.56] | +0.04 | 0.653 | 6.59 ± 0.07 | 1.03× | 10.49 |
| FreeU (strong) | 50 | 26.59 | [25.48, 27.65] | +0.02 | 0.54 | 6.58 ± 0.00 | 1.03× | 10.49 |

## PAG scale (`pag_sweep`, n=50 prompts)

| method | n | CLIPScore | 95% CI | Δ vs base | Wilcoxon p | time (s/img) | rel. time | peak mem (GB) |
|---|---|---|---|---|---|---|---|---|
| SDXL | 50 | 26.57 | [25.72, 27.40] | — | — | 6.39 ± 0.03 | 1.00× | 10.49 |
| PAG s=1 | 50 | 26.11 | [25.30, 26.91] | -0.46 | 0.291 | 9.42 ± 0.00 | 1.48× | 10.49 |
| SDXL+PAG | 50 | 26.17 | [25.21, 27.12] | -0.40 | 0.383 | 9.41 ± 0.01 | 1.47× | 10.48 |
| PAG s=5 | 50 | 25.78 | [24.81, 26.74] | -0.79 | 0.0542 | 9.43 ± 0.01 | 1.48× | 10.49 |

## PAG applied layers (`pag_layers`, n=50 prompts)

| method | n | CLIPScore | 95% CI | Δ vs base | Wilcoxon p | time (s/img) | rel. time | peak mem (GB) |
|---|---|---|---|---|---|---|---|---|
| SDXL | 50 | 26.57 | [25.72, 27.40] | — | — | 6.39 ± 0.03 | 1.00× | 10.49 |
| SDXL+PAG | 50 | 26.17 | [25.21, 27.12] | -0.40 | 0.383 | 9.41 ± 0.01 | 1.47× | 10.48 |
| PAG up_blocks.0 | 50 | 26.16 | [25.00, 27.30] | -0.41 | 0.367 | 9.29 ± 0.04 | 1.46× | 10.49 |

## Human evaluation (3 raters: rater1, rater2, rater3; 60 prompts)

| method | judgments | best rate | worst rate | mean rank (1–4, ↓) |
|---|---|---|---|---|
| SDXL | 180 | 30.0% | 22.2% | 2.38 |
| SDXL+FreeU | 180 | 15.0% | 16.7% | 2.52 |
| SDXL+PAG | 180 | 35.6% | 5.0% | 2.04 |
| FreeU+PAG | 180 | 5.6% | 42.2% | 3.05 |

### Pairwise preference (derived from best/worst ranking)

| pair | A wins | B wins | ties | A win rate (excl. ties) | sign test p |
|---|---|---|---|---|---|
| SDXL vs SDXL+FreeU | 73 | 52 | 55 | 58.4% | 0.0732 |
| SDXL vs SDXL+PAG | 63 | 83 | 34 | 43.2% | 0.116 |
| SDXL vs FreeU+PAG | 87 | 46 | 47 | 65.4% | 0.000477 |
| SDXL+FreeU vs SDXL+PAG | 31 | 77 | 72 | 28.7% | 1.12e-05 |
| SDXL+FreeU vs FreeU+PAG | 96 | 38 | 46 | 71.6% | 5.72e-07 |
| SDXL+PAG vs FreeU+PAG | 114 | 15 | 51 | 88.4% | 5.05e-20 |

Inter-rater agreement on the 'best' choice: Fleiss' κ = 0.105 (over 60 prompts rated by all raters; <0.2 slight, 0.2–0.4 fair, 0.4–0.6 moderate)

### Best-choice rate by rater

| rater | SDXL | SDXL+FreeU | SDXL+PAG | FreeU+PAG |
|---|---|---|---|---|
| rater1 | 58% | 12% | 20% | 3% |
| rater2 | 18% | 18% | 28% | 3% |
| rater3 | 13% | 15% | 58% | 10% |

### Failure tags (count of judgments)

| method | 结构/肢体错误 | 与描述不符 | 物体缺失 | 过饱和/伪影 | 物体重复 | any failure (% judgments) |
|---|---|---|---|---|---|---|
| SDXL | 98 | 26 | 17 | 13 | 12 | 67.8% |
| SDXL+FreeU | 54 | 27 | 13 | 90 | 14 | 76.1% |
| SDXL+PAG | 47 | 25 | 10 | 22 | 16 | 54.4% |
| FreeU+PAG | 48 | 22 | 11 | 147 | 9 | 93.9% |
