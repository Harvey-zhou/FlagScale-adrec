# 14 · 寒武纪 MLU590-M9 适配方案

**状态（2026-11-06）：适配代码和探针已写完，尚未在卡上运行。本文没有任何实测结果；
结果表保持空白，跑完一项填一项。**

- 硬件：MLU590-M9，96 GiB/卡，单节点 8 卡
- 软件：厂商 `torch_mlu 1.34.1+torch2.12.1`，CNCL 通信，Python 3.12.3，
  镜像 `cambricon-base/pytorch:v26.08.0-torch2.12.1-torchmlu1.34.1-ubuntu24.04-py312`
- 范围：BGE-M3、Qwen3-Reranker-8B、CLIP ViT-L/14。Qwen3-Embedding-0.6B 和
  Qwen3-VL-Embedding-8B 不在寒武纪适配范围内
- 目标：**能跑、跑对、可复现**。不为提高指标调模型，也不改共享超参

> **依据的局限**：下面所有"预计可用"的判断来自阅读公开的 torch_mlu `r2.4_develop`
> （2025-03）和 PyTorch v2.12.1 源码，**没有读到集群上实际安装的 1.34.1**。
> 每一项都要由 `probe_mlu.py` 在卡上确认，第 5 节列出了确认方法。

## 1. 路线

走**厂商 torch_mlu + CNCL**，不经 Torch-FL（Torch-FL 的寒武纪后端是另一条线），
参照 Ascend 的 `run-ascend.sh`：

| 项 | 取值 |
|---|---|
| 设备 | `train.hardware.device_type=mlu`，`autocast_device=mlu` |
| 通信 | `distributed_backend=cncl` |
| 平台适配 | `flagscale/platforms/platform_mlu.py`（`import torch_mlu`，不用 `gpu_migration`） |
| 精度 | `master_dtype=fp32`，bf16 计算 |
| FlagGems | `use_flaggems=false`（昆仑芯、PPU 实测都更慢，所有基线统一不开） |
| 超参 | 全部取 `conf/train/<model>.yaml`，命令行只传路径、设备和已登记的平台例外 |

`torch_mlu.utils.gpu_migration` 不使用：它把 `torch.cuda` 改写到 MLU 上，
运行会被记成 `cuda`，等于结果被贴错标签。

### 各模型怎么跑

| 模型 | 卡数 | 并行 | 说明 |
|---|---|---|---|
| BGE-M3 | 1 | 单进程 | 基线（A100 / PPU / 昆仑芯 / Ascend）都是单卡 |
| CLIP ViT-L/14 | 1 | 单进程 | 同上 |
| Qwen3-Reranker-8B | **2** | FSDP2 | batch 32/卡，每步 64 行 = 128 对，与 A100、PPU 同协议 |

**Reranker 为什么先试 2 卡**：8.19B 参数 × 16 字节（fp32 主权重 + 梯度 + AdamW 两个状态）
≈ 131 GB，单卡 96 GiB 放不下，必须分片。A100（80 GB）和 PPU（96 GB）都是 2 卡 FSDP2 跑通的，
PPU 峰值已到 94.9 / 93.9 GB，余量很小。`CLAUDE.md` 和算子清单 xlsx 里的"4 卡起步"是粗估，
先按 2 卡实测，不够再退。

**退路顺序**（每一步都要先写进下面第 7 节的"平台例外"表，再引用结果）：

1. **4 卡、batch 16/卡**：每步仍是 64 行，Ascend 有先例（64 GB 卡）。
   命令：`RERANKER_BATCH=16 bash examples/retrieval/run-cambricon.sh qwen3_reranker 0,1,2,3`。
   这不再是与 A100 同条件的 2 卡对比。
2. **Megatron 流水线**：只作最后手段。`flagscale/train/utils/megatron_pp.py` 直接调
   `torch.cuda.*`（`:68,71,77,517-519`），Megatron-LM-FL 的 `PlatformMLU` 只在
   `release/0.2` 上，且建立在 `gpu_migration` 之上，与本项目规则冲突。
   昆仑芯走 Megatron 是因为 P800 没有 `expandable_segments`，寒武纪有
   （`PYTORCH_MLU_ALLOC_CONF=expandable_segments:True`），所以不应先走这条。

## 2. 代码改动

基线是 wxj630 `feat/retrieval-flagos` 的最新提交（`89f342e`），不是本地旧副本：
旧副本缺 FSDP2 存盘修复 `2e2ef35` / `004cf32`，那两个 bug 的表现是加载不报错、权重却是随机的，
见 [00-index](00-index.md) 的审计一节。

| 文件 | 改动 |
|---|---|
| `flagscale/platforms/platform_mlu.py`（新增，取自 `feat/adrec-cambricon`） | 设备、`cncl`、autocast 类型；PrivateUse1 已被别的后端占用时不导入 `torch_mlu`；新增 `empty_cache` / `memory_stats` |
| `flagscale/platforms/platform_register.py`、`platform_manager.py` | 注册 `mlu`，自动探测顺序 `flagos > cuda > npu > mlu > musa > txda` |
| `flagscale/platforms/platform_base.py` | 新增 `empty_cache()`（默认空操作）和 `memory_stats()`（默认 `None`） |
| `flagscale/platforms/platform_npu.py` | 实现上面两个方法（原先写在训练入口里） |
| `flagscale/train/train_retrieval.py` | FSDP2 分片后的缓存释放和显存日志改走平台接口。原来只有 `device.type == "npu"` 才释放，MLU 不会释放；同时去掉了入口里的厂商设备名 |
| `examples/retrieval/evaluate_retrieval.py` | `--device` 接受已注册的任意原生平台（`mlu`、`npu`），不再只认 `flagos`；平台不可用时抛错，不回退 CPU |
| `examples/retrieval/smoke_model.py` | 设备由 `--device` 指定（默认仍是 `flagos`），不再写死 `flagos:0` |
| `examples/retrieval/measure_throughput.py`、`profile_step_segments.py` | 同步函数加入 `mlu`；profiler 在 MLU 上用 `ProfilerActivity.PrivateUse1` |
| `examples/retrieval/conf/train.yaml` | `WANDB_MODE` 改为 `${oc.env:WANDB_MODE,online}`，默认值不变，无外网的机器可设 `offline` |
| `examples/retrieval/run-cambricon.sh`（新增） | 启动脚本，见第 4 节 |
| `examples/retrieval/cambricon-container.sh`（新增） | 容器封装，取自 ads/rec 分支，挂载检索模型和数据目录 |
| `examples/retrieval/probe_mlu.py`（新增） | 栈探针，见第 5 节 |

模型、损失、数据三层没有任何厂商设备名，也没有改动。

## 3. 环境准备

集群上已确认的（来自 ads/rec 分支的 `examples/CAMBRICON.md`，2026-11-03）：8 × MLU590-M9、
驱动 6.5.52、镜像已加载、`/srv` 有 5.9 TB 空闲、`hf-mirror.com` 可用、
**卡时钟被限制在 1200 MHz（默认 1600）**。

**没有确认的**，开始前要先查：

| 项 | 怎么查 |
|---|---|
| 三个模型（`BAAI/bge-m3`、`openai/clip-vit-large-patch14`、`Qwen/Qwen3-Reranker-8B`）在不在集群上 | `run-cambricon.sh` 的预检会逐个检查配置里的 `model.model_path` |
| T²-Ranking `triplet-15`（11 个 parquet、340,111 行）和 `wds_mscoco_captions2017` 在不在 | 同上；文档没有记录 triplet-15 的来源，可能要从别的机器拷 |
| 镜像里有没有 `transformers`、`tokenizers`、`safetensors`、`Pillow`、`pyarrow`、`webdataset` | 预检会列出缺失的包 |
| PyPI 能否访问 | 不能时从其他机器下 wheel |

依赖用 `pip install --target <目录>` 装到独立目录，再设 `RETRIEVAL_EXTRA_PYPATH=<目录>`，
**不覆盖厂商的 torch 和 torch_mlu**。算子清单 xlsx 记的 transformers 版本是 4.57.6。
Ascend 曾遇到 pyarrow 26 需要 NumPy 2，装完要核对 NumPy 兼容性。

模型和数据在宿主机上的位置不固定，配置里写的是 `/flagos-retrieval-models/...` 和
`/flagos-retrieval-datasets/...`。用容器时：

```bash
export CAMBRICON_IMAGE=cambricon-base/pytorch:v26.08.0-torch2.12.1-torchmlu1.34.1-ubuntu24.04-py312
export WORK_ROOT=/srv/$USER
export RETRIEVAL_MODELS_HOST=<含 bge-m3/ clip-vit-large-patch14/ Qwen3-Reranker-8B/ 的目录>
export RETRIEVAL_DATA_HOST=<含 t2ranking/ wds_mscoco_captions2017/ 的目录>
bash examples/retrieval/cambricon-container.sh bash examples/retrieval/run-cambricon.sh bge_m3 0
```

目录结构要和配置里的路径对上（例如 `bge-m3/bge-m3`、`t2ranking/t2ranking/triplet-15`）。

## 4. 启动脚本 `run-cambricon.sh`

```bash
bash examples/retrieval/run-cambricon.sh bge_m3 0
bash examples/retrieval/run-cambricon.sh clip_vit_large 1
bash examples/retrieval/run-cambricon.sh qwen3_reranker 0,1
PREFLIGHT_ONLY=1 bash examples/retrieval/run-cambricon.sh qwen3_reranker 0,1   # 只检查，不训练
SMOKE=1 bash examples/retrieval/run-cambricon.sh clip_vit_large 0              # 一个真实 batch
SMOKE=1 ALLOW_CPU_FALLBACK=1 bash examples/retrieval/run-cambricon.sh bge_m3 0 # 一次列出所有会走 CPU 的算子
```

脚本做的事（都是为了不让一次假成功混进结果）：

| 防护 | 做法 |
|---|---|
| 不静默回退 CPU | `ENABLE_FALLBACK_TO_CPU=0`；另外扫日志里的 `will fall back to run on the CPU`，命中即判失败。不依赖变量名在 1.34.1 里是否改过 |
| 只用点名的卡 | 设 `MLU_VISIBLE_DEVICES`；探针核对可见卡数等于进程数 |
| 平台固定 | 脚本里固定 `FLAGSCALE_DEVICE=mlu`、`FS_PLATFORM=mlu`，shell 里残留的值不会改掉它 |
| 先检查环境 | 缺包、模型或数据路径不存在，在加载模型之前就停 |
| 先跑探针 | 探针不过不开始训练（`ALLOW_CPU_FALLBACK=1` 的探测模式除外） |
| 记录时钟 | 把 `cnmon` 输出存进预检日志，每个吞吐数字都要配时钟 |
| 显存分配器 | 默认 `PYTORCH_MLU_ALLOC_CONF=expandable_segments:True` |
| 输出位置 | 默认 `/srv/$USER/flagos-retrieval-cambricon/<模型>`，不写小根分区 |

`before_start` 被替换成 `true`，由脚本自己把环境放进 `PATH`，所以不需要宿主机上存在
`conf/train.yaml` 默认的 `/flagos-search-codes/...` 环境。

## 5. 探针：每项要在卡上确认什么

`examples/retrieval/probe_mlu.py`，单卡跑通用检查，2 卡以上再加分布式检查。
下面每一项都对应一个**源码里看不出结论、或结论会悄悄出错**的风险：

| 风险 | 为什么要查 | 探针怎么判 |
|---|---|---|
| **autocast 悄悄失效** | BGE、CLIP 的 bf16 只来自 autocast。dtype 不被支持时 PyTorch 只警告并关闭 autocast，而 `train_retrieval.py` 的 `autocast_context` 还会吞掉 `RuntimeError` / `TypeError`，结果是整轮 fp32 且不报错 | `matmul`、`linear`、`bmm` 在 `torch.autocast("mlu", dtype=torch.bfloat16)` 下输出必须是 bf16 |
| **float64 all_reduce 算错** | `reduce_metrics` 用 float64 汇总每步 loss。公开版 torch_mlu 把 `kDouble` 映射成 `cnclFloat`（`process_group_cncl.cpp:202`）；集群预检说 fp64 可用，但那是单卡张量运算，不是通信。映射没变时汇报的 loss 会错且不报错 | 各卡贡献 `[1.5, 1+2⁻⁴⁰]`，求和后第二项必须保留 2⁻⁴⁰（fp32 表示不了） |
| **FSDP2 内部算子缺 MLU kernel** | 公开 yaml 里 `_chunk_cat`、`split_with_sizes_copy.out` 没有 MLU kernel，走 PyTorch 的组合实现。预计不回退 CPU，但要在关闭回退的严格模式下确认，也可能偏慢 | 单卡直接调用这些算子，与 CPU 结果比对 |
| **CNCL 缺 coalesced 方法** | FSDP2 要 `allgather_into_tensor_coalesced` 等。PPU、昆仑芯、Ascend 都缺过，所以 `parallel.py` 里有垫片；公开版 CNCL 有这些方法，1.34.1 不一定 | 检查方法是否存在（WARN 级）；真正的判据是下一行 |
| **FSDP2 本身** | 厂商测试里没有 `fully_shard` 的用例，torch_mlu 里的 FSDP 代码是 FSDP1 的初始化补丁 | 用 `apply_fsdp2` 分片一个小模型，fp32 主权重 + bf16 计算，首步 loss 与 fp32 CPU 参考比对，训练 3 步 loss 下降，`full_state_dict` 的往返结果与初始权重逐元素相等 |
| **多流、事件** | FSDP2 在计算流和通信流之间同步 | `torch.mlu.Stream` / `Event` / `current_stream` 能建能用 |
| **bf16 数值** | 确认 bf16 matmul 和 SDPA（带 bool mask）与 CPU 一致 | 相对误差 < 2e-2；SDPA 最大绝对误差 < 5e-2 |
| **页锁定内存** | DataLoader 的 `pin_memory` | `pin_memory()` 后拷到卡上 |

探针通过只说明**这些点**没问题，不代表模型能跑通，更不代表吞吐达标。

## 6. 上卡顺序

1. **环境和数据**：`tools/audit_retrieval.py`、`tools/verify_split.py`，
   划分应打印 `train=306806 validation=16565 test=16740`。
2. **探针**：先单卡，再 `0,1` 两卡（`PREFLIGHT_ONLY=1`）。float64 和 FSDP2 的结论在这一步出。
3. **逐模型算子探测**：CLIP → BGE → Reranker。先 `SMOKE=1 ALLOW_CPU_FALLBACK=1` 一次列出缺口，
   再去掉 `ALLOW_CPU_FALLBACK` 严格跑通。
4. **2 卡 FSDP2 先用 CLIP 验**：与单卡对比前几步 loss，再做存盘往返。
5. **Reranker 小样本 2 卡**：用 `train.data.max_samples` 限样本，记录显存峰值，
   存盘 → 重新加载 → 评测。**存盘必须在长跑之前验**：
   `input_layernorm.weight` 不能全是 1.0，配对准确率不能低于 base
   （A100 为 83.8%，base 83.2%）。FSDP2 存盘的旧 bug 就是这样悄悄坏掉的。
6. **完整训练**（只是按其他芯片的吞吐推的预算，不是预测）：

   | 模型 | 卡数 | 预计耗时 | 依据 |
   |---|---|---|---|
   | CLIP | 1 | 约 1 小时（0.74–1.28 h） | A100 119.0、PPU 69.1 样本/s 之间 |
   | BGE-M3 | 1 | 约 11–24 小时 | 同上，BGE 区间 |
   | Reranker | 2 | 约 67–131 小时 | 1,840,836 对 ÷ 7.62–3.9 对/s |

   三个模型同时跑要占 4 张卡，要和同机的 ads/rec 长跑错开。
7. **评测**：`EVAL_DEVICE=mlu`，Reranker 用 `NUM_NEGATIVES=15`、`BATCH_SIZE=8`，
   再 `tools/recompute_metrics.py --ndcg`，与 [07](07-a100.md) 对比。
8. **吞吐不达标时**：用 `profile_step_segments.py` 按真实整步归因，不要按微基准排优先级
   （[08](08-kunlunxin.md) 里微基准和小 batch 外推给过错误排序）。

## 7. 验收口径

沿用 [10-microbench-discount.md](10-microbench-discount.md)：吞吐阈值 =
A100 × 折扣系数 × 0.8，精度 ≥ A100 的 95%。寒武纪折扣系数 1.215（不含 FP32 口径），
所以阈值是 A100 的 **97.2%**：

| 模型 | A100 | 阈值 | 昆仑芯 / PPU 占 A100 |
|---|---|---|---|
| BGE-M3（样本/s，单卡） | 118.0 | **114.7** | 46% / 57% |
| CLIP（样本/s，单卡） | 119.0 | **115.7** | 70% / 58% |
| Reranker（对/s，2 卡） | 7.62 | **7.41** | 83% / 51% |

微基准（10 号文档）里 MLU590 的 bf16 是 291.96 TFLOPS（A100 的 110%），带宽 143%，
**文档没写这个微基准是否在 1200 MHz 限频下测的**。如果不是，系数和实测吞吐就不在同一条件，
需要向测微基准的人确认。集群当前限频 1200 MHz（默认 1600），吞吐很可能因此偏低。
"跑通"和"达标"要分开汇报；Reranker 跨平台只比对/s，不比 s/步。

### 平台例外登记表

只有这张表里写了的偏差，才允许出现在命令行上。

| 日期 | 项 | 取值 | 原因 | 影响 |
|---|---|---|---|---|
| — | — | — | — | 暂无；2 卡 batch 32 与共享配置一致 |

`lr 1e-7`、`skip_nonfinite`、`XMLIR_BMM_DISPATCH_VALUE` 是 P800 专属，不带过来；
先用共享配置跑。

## 8. 结果

**以下全部待填，没有跑过。**

### 栈探针

| 检查项 | 结果 | 日期 | 日志 |
|---|---|---|---|
| autocast 输出 bf16 | TODO | | |
| float64 all_reduce 精确 | TODO | | |
| FSDP2 内部算子（严格模式） | TODO | | |
| CNCL coalesced 方法 | TODO | | |
| 小模型 FSDP2 往返 | TODO | | |

### 训练与评测

| 模型 | 卡数 | 时钟 | 吞吐 | 占 A100 | 显存峰值 | 精度（占 A100） | 备注 |
|---|---|---|---|---|---|---|---|
| BGE-M3 | 1 | TODO | TODO | | | | |
| CLIP | 1 | TODO | TODO | | | | |
| Reranker | 2 | TODO | TODO | | | | |

## 9. 已知的未决问题

- DDP 多卡路径：`train_retrieval.py` 只给 `{"cuda","npu","musa"}` 传 `device_ids`。
  BGE、CLIP 都是单卡（`world_size=1`），不走这条，所以没有加 `mlu`；
  以后要在 MLU 上跑 DDP 多卡，先验证 torch_mlu 的 DDP 是否需要 `device_ids`。
- `megatron_pp.py` 里的 `torch.cuda.*` 调用不在本方案范围内，见第 1 节的退路说明。
- Torch-FL 路线（`device_type=flagos`）不在本方案内。它的寒武纪后端 2026-11-04 起已存在，
  但对检索三模型仍有缺口（CLIP 的卷积反向会报错、dtype 转换经主机、`index.Tensor` / `mul_.Tensor` 回退 CPU、
  通信没有可用路径），详见 [15-torch-fl-cambricon-gaps.md](15-torch-fl-cambricon-gaps.md)。
  `CLAUDE.md` 与算子清单 xlsx 里"`device_type=flagos`"的写法描述的是那条路线。
