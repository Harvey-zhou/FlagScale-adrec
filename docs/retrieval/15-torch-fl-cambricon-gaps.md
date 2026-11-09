# 15 · Torch-FL 寒武纪适配：对检索和广推模型的缺口

**状态（2026-11-09）：全部来自读源码，没有在卡上运行任何东西。** 每条结论都标了依据，
带“推断”的是读源码推出来的、还要在卡上确认。

- 被检查的对象：[flagos-ai/Torch-FL](https://github.com/flagos-ai/Torch-FL) `main`，提交 `9613cfd`（2026-11-08）
- 对照：PyTorch `v2.10.0`（Torch-FL 钉的版本）的 `native_functions.yaml` 和 FSDP2 源码
- 结论一句话：**Torch-FL 的寒武纪后端已经存在，但还撑不起检索三模型和广推六模型的训练**；
  本仓库继续以 torch_mlu + CNCL 为主路线（[14-cambricon.md](14-cambricon.md)），Torch-FL 作对照路线

## 1. 它是什么

| 项 | 内容 |
|---|---|
| 加入时间 | #720，2026-11-04；之后的算子优化：#736、#748/#749、#754、#757、#767–#772 |
| 形态 | flagos 设备上的**原生后端**，直接调 libcnrt（`csrc/runtime/accelerator/cambricon/`） |
| torch | 普通的 `torch 2.10.0+cpu`；**不依赖 torch_mlu**，`torch.mlu` / `torch_mlu` 由 `_cambricon_compat.py` 在导入时合成 |
| Triton | `triton 3.4.0+mlu2.1.1`；没有寒武纪的 flagtree 包 |
| 算子路径 | FlagGems 为主，加一小批 CNNL 原生 kernel |
| 发布 | 没有发布包。CI 通道因无 runner 退役，`pypi_lane` 为空，要从源码编译，需要 Neuware SDK（上游测试用 4.6.7～4.7.6） |
| 与 torch_mlu 的关系 | **互斥**：两者都占 PrivateUse1 槽位；集群镜像是 `torch 2.12.1 + torch_mlu 1.34.1`，Torch-FL 需要 2.10.0，要单独建环境 |

上游自己的测量：fp32、单卡、4 层 hidden 1024 的小 Qwen3，batch 2、seq 512，前向 30 ms（torch_mlu 17.7 ms）。
**没有 bf16、8B、多卡的测量。**

## 2. 怎么读“算子路由”

`backends_cambricon.conf` 把 2042 个算子分三类：**645 个有加速实现**（616 FlagGems + 29 CNNL 原生），
**1397 个是 `none`**。

`none` **不等于回退 CPU**。PyTorch 对很多算子给了复合实现，会拆成别的算子在设备上跑
（`stack`、`_foreach_*`、`_chunk_cat.out`、`split_with_sizes_copy.out`、`binary_cross_entropy_with_logits`、
`linear`、`matmul`、`unbind` 等）；只有**没有复合实现的原生算子**才进 `cpu_fallback`。
判断方法：对照 PyTorch 2.10 的 `native_functions.yaml`。条目的分发里有 `CompositeExplicitAutograd`，或者没有 `dispatch`
也没有 `structured_delegate`（即隐式复合），就会拆开；只有各后端自己的 kernel（CPU、CUDA、MPS……）或结构化 kernel 的才是原生算子。

另有两点要记住：

- **CPU 回退默认是静默的**：只有设了 `FLAGOS_LOG=fallback`（`FLAGOS_LOG_FALLBACK=1`）才会打印
  `[flagos cpu_fallback] aten::<op>`，**没有“遇到就报错”的开关**。本项目“不静默回退”的要求，只能靠扫日志满足。
- **回退不一定是“慢但对”**：个别算子（卷积反向）没有可回退的实现，会直接抛异常，见下。

## 3. 对检索三模型的缺口

| 缺口 | 依据 | 影响 |
|---|---|---|
| **CLIP 的 `Conv2d` 反向会报错** | `convolution_backward_overrideable` 路由为 `none`；`register.cc:1007-1016` 的卷积补位名单里没有寒武纪（TsingMicro、CUDA、DCU 补前向和反向，MetaX、PPU 补反向）；ATen 自带的唯一实现会抛异常 | **CLIP 无法训练**（patch embedding 是 Conv2d 且要更新）。前向走 FlagGems 的 `convolution_overrideable`。报错这一条是推断 |
| **dtype 转换仍走“设备→主机→设备”** | `copy_ops.cc:564-572` 寒武纪分支 `copied_on_device = false`；#769 只把跨步张量的 `contiguous()` 和 `clone` 搬到了设备上 | 每次 fp32↔bf16 转换都过一遍主机：autocast 对权重的转换（BGE、CLIP）、HF RMSNorm 里的 `.to(float32)`（Reranker 每层 2 次）、FSDP2 每次 all-gather 前把 fp32 分片转成 bf16（`_foreach_copy_` 复合拆成逐张量 `copy_`）。**推断**为吞吐的头号杀手 |
| **`index.Tensor` 无实现** | 路由 `none`，原生结构化算子 | Reranker 的 `logits[rows, positions]`：配置 `use_score_head: false`，`logits` 形状 `[batch, seq, 151669]`，按 batch 32、seq 640、bf16 估算约 6 GB，**每步前向整块经主机**；反向的 `_index_put_impl_` 有共享包装（`register.cc:636`），是否在设备上执行要实测 |
| **`mul_.Tensor` 无实现** | 路由 `none`，原生结构化算子 | `torch.nn.utils.clip_grad_norm_`（三个模型都开，`max_grad_norm: 1.0`）经 `_foreach_mul_.Tensor` 对**每个参数的梯度**触发；`binary_cross_entropy_with_logits`（Reranker 的损失）内部 `(1 - target).mul_(input)` 也触发 |
| **`native_layer_norm_backward` 无实现** | 路由 `none`，原生算子 | BGE-M3、CLIP 的每个 LayerNorm 反向走 CPU。Qwen3 用手写 RMSNorm，不涉及 |
| **融合注意力不一定生效** | `sdp_choice_stub.cc`：只服务 half/bf16、**无 attn_mask**、无 dropout、无 GQA 扩展；`attention.cc` 把 `attn_mask` 固定传空 | 检索的 batch 带 padding mask（BGE 双向，Qwen3 因果 + padding），Qwen3-8B 是 GQA。融合路径大概率不触发，改走 FlagGems/分解路径。推断 |
| **通信只有 FlagCX 一条路** | `process_group.py:184` 的 `_VendorProfile("mlu", None, None)`：`flagcx_native=False`、`direct=False`、无 view，选中 FlagCX 会在 `_resolve_view` 抛 `NotImplementedError`；FlagCX 无 neuware 4.6.7 的发布包；FlagCX 不可用时退到主机中转的 gloo | Reranker 的 2 卡 FSDP2 在通信层没有可用路径；没有任何寒武纪多卡或 FSDP2 的测试与文档 |
| **FlagGems 固定开销** | #770 实测：留在 FlagGems 上的算子每次调用约 1.2 ms，与张量大小无关 | `addmm`、`native_layer_norm`、`gelu`、`_softmax`、`native_batch_norm` 等仍在 FlagGems 上，一个 8B 模型一步的算子调用数很大 |
| `torch.compile` | #772：整模型仍编译失败 | 不能靠编译补算子缺口 |

**反过来，不是缺口的**（早先估计偏悲观，这里更正）：

- `_chunk_cat.out`、`split_with_sizes_copy.out`、`_foreach_copy_`、`_foreach_norm`、`_foreach_*`（AdamW）
  在 PyTorch 里是 `CompositeExplicitAutograd`，会拆成逐张量算子在设备上跑，**不会回退 CPU**，只是比融合内核慢。
- FSDP2 的三个自定义算子 `all_gather_copy_in`、`split_with_sizes_copy`、`chunk_cat` 在 PyTorch 2.10
  源码里都注册了 `PrivateUse1`（`_fsdp_collectives.py:174,202,225`）。
- `embedding` 前向和 `embedding_dense_backward`、`mm`、`bmm`、`sum`、`mean`、`pow.Tensor_Scalar`、`rsqrt`、
  `silu`、`cat`、`cumsum` 是 CNNL 原生；`softmax`、`log_softmax`、`nll_loss`（交叉熵）、`gelu`、
  `native_layer_norm` 前向在 FlagGems 上有路由。

## 4. 对广推六模型的缺口

按 `flagscale/models/{ads,rec}`、`flagscale/train/losses/{ads,rec}.py`、`train_ads.py`、`train_rec.py` 逐个算子对照。
单卡、fp32 的场景。

| 缺口 | 影响的模型 | 后果 |
|---|---|---|
| `mul_.Tensor`（回退 CPU） | DIN、DeepFM、DCN-Mix、DSSM、SASRec | 它们的损失都用 `binary_cross_entropy_with_logits`，内部有 `(1 - target).mul_(input)`，**每步前向都触发**；`train_ads.py` 的 `clip_grad_norm_` 还会对每个参数触发 |
| `native_layer_norm_backward`（回退 CPU） | SASRec | LayerNorm 反向 |
| `_prelu_kernel_backward`（回退 CPU） | DSSM；DIN 用 PReLU 激活时 | PReLU 反向 |
| `index.Tensor`、`index_select`（回退 CPU） | LightGCN | `users[batch["user"]]` 取值，以及 `index_add` 传播路径 |
| 卷积反向（**报错**） | SASRec 的 `Conv1d(kernel_size=1)` | 同第 3 节 CLIP 的原因，训练跑不了。推断 |
| 无稀疏后端 | LightGCN 默认路径 | `torch.sparse.mm` 不可用，继续用 `propagation=index_add` |
| 带 dtype 的 `copy_` 经主机 | 全部 | 掩码 `.to(float)`、标签 `.to(dtype)`，张量小，影响有限 |
| FlagGems 固定开销 | 全部 | 模型小，一步的时间可能被每次 ~1.2 ms 的调用开销主导 |

不是缺口的：embedding 前向/反向、`mm`/`bmm`、sum/mean、softmax、BatchNorm 前向和反向、dropout、
`log_sigmoid` 前向和反向、softplus、Adam（`_foreach_*` 复合拆分）都有路径。

**对照**：同样的 DIN、DeepFM、DCN-Mix，在 torch_mlu 路线上严格模式（`ENABLE_FALLBACK_TO_CPU=0`）的
3 步探针已通过，没有算子走 CPU（ads/rec 分支的 `examples/CAMBRICON.md`）。

## 5. 上卡后怎么确认

先不需要整套训练，一步就够：

```bash
# 在 Torch-FL 的寒武纪环境里（torch 2.10.0+cpu，从源码编译的 torch_fl）
export FLAGOS_LOG=fallback FLAGSCALE_DEVICE=flagos FS_PLATFORM=flagos
python examples/retrieval/smoke_model.py clip_vit_large --device flagos 2>&1 | tee smoke.log
grep -o '\[flagos cpu_fallback\] [^ ]*' smoke.log | sort | uniq -c | sort -rn
```

- `smoke_model.py` 只做一个 batch 的前向、反向和一次 AdamW，能同时暴露卷积反向的报错和回退算子的清单。
  依次跑 `clip_vit_large`、`qwen3_reranker`。注意 `smoke_model.py` 里 Reranker 的配置是 `freeze_backbone` + `use_score_head`，
  索引的是 `hidden` 而不是训练配置里的 `logits`，所以它暴露不出 6 GB 那一条，也不涉及 FSDP2 和通信。
- 广推：在 ads/rec 分支用 `examples/ads/smoke.py`、`examples/rec/smoke.py`，加 `--device flagos`（命令见那个仓库的 `CLAUDE.md` “运行方式”），同样配 `FLAGOS_LOG=fallback`。
- 对本节第 3、4 节里每条标“推断”的，以这份清单为准更新本文。

## 6. 什么时候值得再看一遍

Torch-FL 更新很快（一周内合入十几个寒武纪提交）。重新评估前先看这几项有没有变化：

1. `backends_cambricon.conf` 里 `convolution_backward_overrideable`、`native_layer_norm_backward`、`mul_.Tensor`、
   `index.Tensor`、`_prelu_kernel_backward` 是否有了实现（有就从上面的表里划掉）。
2. `copy_ops.cc` 的寒武纪分支是否有了设备上的 dtype 转换。
3. `process_group.py` 里寒武纪的画像是否有了 `flagcx_native=True` 或 `direct=True`，FlagCX 是否发布了对应 Neuware 版本。
4. 是否出现 bf16、8B 或多卡的实测，以及 `.github/configs/cambricon.yml`（有 runner 的信号）。
