# Training-free Diffusion Enhancement: SDXL vs FreeU vs PAG

在不重新训练模型的前提下，比较原始 SDXL、SDXL + FreeU、SDXL + PAG 的生成质量、文本一致性和推理开销。

## 目录结构

```
configs/experiments.yaml   所有实验参数（唯一需要改的配置文件）
scripts/01_sample_prompts.py   从 COCO val2017 captions 分层抽样 prompt + 固定 seed
scripts/02_generate.py         生成图片，记录时间与显存（支持断点续跑）
scripts/03_clipscore.py        计算 CLIPScore
scripts/04_build_blind_eval.py 打包人工盲评网页
scripts/05_analyze.py          汇总表格 + 图表 + 统计检验
scripts/06_make_grids.py       并排对比图（自动挑选成功/失败案例）
scripts/07_report_figures.py   报告用英文图表
scripts/08_clip_similarity.py  与原始 SDXL 图片的 CLIP 图像相似度
src/dune.py                    DUNE（Lim et al., 2026）检测阶段的复现
presentation/                  8 分钟展示 slides（.pptx/.pdf，含演讲备注）及生成脚本 build_deck.js
report/report_en.docx          英文完整报告（Word）
report/report_zh.docx          中文完整报告（Word，英文版的翻译）
slurm/dune.sh                  集群上跑 DUNE + 相似度
src/blind_eval.html            盲评网页模板
colab_run.ipynb                Colab 一键运行笔记本
```

运行后产生：

```
data/prompts.csv           抽样得到的 prompt、类别、seed、是否属于子集
outputs/images/<run>/      每组实验的图片
outputs/records.csv        每张图的生成时间、峰值显存、参数
outputs/env.json           GPU / torch / diffusers 版本（写进报告）
results/                   clipscore.csv、summary.md、table_*.csv、fig_*.png、grids/
blind_eval/                发给评审的盲评网页（可直接压缩分享）
blind_eval_key.json        A/B/C 与方法的对应关系（不要发给评审！）
eval_results/              收回的 results_<名字>.json 放这里
```

## 实验设计

| run | 用途 | prompt | FreeU | PAG |
|---|---|---|---|---|
| baseline | 主实验 | 全部 | 关 | 0 |
| freeu | 主实验 | 全部 | b1=1.3 b2=1.4 s1=0.9 s2=0.2（官方 SDXL 推荐） | 0 |
| pag | 主实验 | 全部 | 关 | 3.0，mid 层 |
| freeu_pag | 主实验（第四组：叠加） | 全部 | 官方值 | 3.0，mid 层 |
| dune | 主实验（第五组：DUNE 复现，见 src/dune.py） | 全部 | 关 | 0（DUNE κ=0.3, γ=0.7, p=0.9） |
| freeu_weak / freeu_strong | FreeU 超参 | 子集 | 更弱 / 更强 | 0 |
| pag_s1 / pag_s5 | PAG scale | 子集 | 关 | 1 / 5 |
| pag_up | PAG 作用层 | 子集 | 关 | 3，up_blocks.0 |

固定参数：SDXL base 1.0、1024×1024、30 步、CFG 5.0、Euler scheduler、fp16、每条 prompt 固定 seed。

> 实现说明：所有 run 共用同一个 `StableDiffusionXLPAGPipeline`。`pag_scale=0` 时它与原始 SDXL pipeline 完全一致，FreeU 通过 `enable_freeu()` / `disable_freeu()` 开关，因此三种方法唯一的区别就是被比较的那个组件。

## 运行步骤

推荐在 Colab（A100 / L4）上运行 `colab_run.ipynb`。命令行版本：

```bash
pip install -r requirements.txt

# 1. 抽 150 条 prompt，其中 50 条作为超参实验子集（首次会下载约 240MB 的 COCO 标注）
python scripts/01_sample_prompts.py --n 150 --subset 50

# 2. 先做冒烟测试：每组只跑 5 张，确认环境没问题
python scripts/02_generate.py --group main --limit 5

# 3. 主实验 + 超参实验（断线后重跑同一命令会自动续跑）
python scripts/02_generate.py --group main
python scripts/02_generate.py --group freeu_sweep --group pag_sweep
python scripts/02_generate.py --runs pag_up freeu_pag        # 可选

# 4. CLIPScore
python scripts/03_clipscore.py

# 5. 打包盲评 -> 三人分别评分 -> 收回 JSON 放进 eval_results/
python scripts/04_build_blind_eval.py

# 6. 汇总分析与作图
python scripts/05_analyze.py
python scripts/06_make_grids.py
```

不想等 GPU、只想先熟悉流程？所有步骤都可以加 `--dry-run`（生成占位图和随机分数），几秒就能把整条流程走通。dry-run 的结果会在 summary.md 顶部标出警告，**不要**拿来写报告。正式运行前删掉 `outputs/`、`results/clipscore.csv`、`blind_eval*`、`eval_results/`。

### 在 NTU CCDS TC2 集群上运行（Slurm）

账号限额：1 块 A40（46GB）、≤10 CPU、≤30GB 内存、单个作业 ≤6 小时，同时最多 2 个作业。登录节点不能跑代码，计算一律用 `sbatch` 提交。

```bash
# 一次性：在 home 目录建 conda 环境（登录节点上安装环境是手册允许的）
module load miniconda3 && eval "$(conda shell.bash hook)"
conda create -y -n sdxl python=3.11 && conda activate sdxl
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt

# 在工程根目录提交
cd ~/sdxl-freeu-pag
sbatch slurm/smoke.sh      # 下载模型 + 抽 prompt + 每组 5 张
sbatch slurm/run_all.sh    # 全部实验 + CLIPScore + 分析；超时中断后重新提交会续跑
squeue -u $USER            # 查看排队/运行状态
tail -f slurm/logs/sdxl_all_<jobid>.out
```

结果拉回本地（在 Mac 上运行）：

```bash
rsync -az <用户名>@<集群登录节点>:sdxl-freeu-pag/results/ ./results/
```

### 算力估计（粗略）

总计约 450 张（主实验）+ 300 张（子集实验）= 750 张，PAG 每张约为 baseline 的 1.5 倍时间。

| GPU | baseline 每张 | 全部实验 |
|---|---|---|
| A100 | ~4–5 s | ~1–1.5 小时 |
| L4 | ~12–15 s | ~3–4 小时 |
| T4 | ~35–45 s | ~9–10 小时（建议减少 prompt 或分几次跑） |

显存不够（<12GB）时加 `--cpu-offload`，但计时会变慢，报告里要说明。

## 在线盲评（推荐）

`scripts/04b_build_online_eval.py` 把离线盲评包做成一个网页，发布为 claude.ai Artifact：
<https://claude.ai/artifact/D5aErmVqN1b61XQfFYKx7o>

- 评审打开链接、输入名字即可开始，不用下载或解压；每一步都自动保存到共享数据库，不用导出 JSON。
- 点一张图选中它，再按 1–5 给它打失败类型标签；选“最差”时会自动选中那张图。
- 数据库规则：每位评审只能读写自己的 `ratings/<id>`，彼此看不到；只有页面所有者能读全部结果。
- 页面所有者需在 Share 菜单里把链接分享给队友，并给他们“可编辑 / Contributor 以上”的权限，否则他们无法保存。
- 评完后让 Claude 读取数据库，写成 `eval_results/results_<名字>.json`，再运行 `05_analyze.py`。
- 在线版和离线版用同一份 `blind_eval_key.json`，两种方式收回的结果可以混用。

## 自托管盲评（放在自己的服务器上）

队友没有 claude.ai 账号时，可以把同一个页面放到自己的云服务器上，评分存在服务器的 JSON 文件里。服务端 `scripts/eval_server.py` 只用 Python 标准库。

```bash
python scripts/04b_build_online_eval.py --server            # 生成 online_eval_server/
rsync -az online_eval_server/ ubuntu@<服务器>:sdxl-eval/site/
rsync -az scripts/eval_server.py ubuntu@<服务器>:sdxl-eval/
# 服务器上（建议用 systemd 托管，见 sdxl-eval.service）：
python3 eval_server.py --site site --results results --port 8080 --token <随机口令>
```

- 访问地址是 `http://<服务器>:8080/<随机口令>/`，没有口令的请求一律 404；云服务商的防火墙或安全组要放行 TCP 8080。
- 答案映射 `blind_eval_key.json` 不放到服务器上。
- 每位评审的结果存为服务器上的 `sdxl-eval/results/results_<名字>.json`，格式与离线版导出相同。评完后拉回本地：
  `rsync -az ubuntu@<服务器>:sdxl-eval/results/ eval_results/`，再运行 `05_analyze.py`。

## 人工盲评流程（三人）

1. 负责打包的同学运行 `04_build_blind_eval.py`，**不要打开** `blind_eval_key.json`。
2. 把 `blind_eval/` 文件夹压缩发给三个人（包括自己）。
3. 每人解压后双击 `index.html`，输入自己的名字开始评分：
   - 每条 prompt 选出 **最好** 和 **最差** 的一张（实在分不出时选“都差不多”）；
   - 可选：给有问题的图片勾选失败类型（物体缺失、结构错误、重复、过饱和/伪影、与描述不符）；
   - 快捷键 Q/W/E/R 选最好，A/S/D/F 选最差，T 差不多，←/→ 翻页；进度自动保存。
4. 完成后点“导出结果 JSON”，把 `results_<名字>.json` 发回，放进 `eval_results/`。
5. 运行 `05_analyze.py` 会得到：最好率 / 最差率 / 平均名次、两两胜率（符号检验）、三人一致性 Fleiss' κ、每人偏好、失败类型统计。

每条 prompt 选出最好和最差，就得到了每组图片的部分排序（最好第 1、最差第 4、其余两张并列 2.5），可以由此推出方案中要求的 pairwise preference。评审本人也是作者，可能有偏差，这一点要写进 Limitations；如果能再请 1–2 位同学参与，工具支持任意人数。

## 三人分工建议

| 成员 | 工程任务 | 报告 | Presentation |
|---|---|---|---|
| A：工程 + 生成 | 搭 GPU 环境、跑 01/02、维护 outputs、记录 env.json | Experimental setup、计算开销分析 | 背景 + SDXL（0:00–1:45）、实验设置（4:00–5:00） |
| B：FreeU + 分析 | 精读 FreeU、FreeU 超参实验分析、跑 03/05、整理表格 | Methods-FreeU、主结果、FreeU 分析 | FreeU（1:45–2:45）、结果（5:00–6:45） |
| C：PAG + 盲评 + 案例 | 精读 PAG、PAG scale/layer 分析、跑 04 打包盲评、跑 06 挑案例 | Methods-PAG、PAG 分析、成功/失败案例 | PAG（2:45–4:00）、失败案例与结论（6:45–8:00） |

三人共同：人工盲评、Introduction / Conclusion、统稿、计时彩排。

## 报告里需要说明的局限

- prompt 只有 100–200 条 COCO caption，结论不一定能推广到艺术风格或长 prompt。
- CLIPScore 对细节质量不敏感，主要反映语义一致性；人工评价人数少且是作者本人。
- 只测试了一个 scheduler、一种步数和一个 CFG 值；FreeU / PAG 的最优参数可能随这些设置变化。
- 没有使用 refiner。
