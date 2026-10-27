# ST CS 依赖与验证入口

本分支将 on-demand 生命周期迁移到 `dev-swift-v5`，并保留 `model_cls` / `model_loader` 从 YAML、训练目录、持久化配置到 worker 的链路。
ST 使用 Swift 的 `MultiLoraSentenceTransformerModel`，依赖固定在 `requirements-st.txt` 的完整 SHA；不另建训练后端。

## 固定版本的梯度检查点限制

Swift `7e4203a961416952e1c8ea5b8fb1763139d56bfa` 的多租户 ST 构造函数调用：

```python
self._enable_gradient_checkpointing(gradient_checkpointing_kwargs)
```

其父类 `SentenceTransformerModel._enable_gradient_checkpointing(self)` 不接受该参数，因此默认构造在创建租户前失败：

```text
TypeError: SentenceTransformerModel._enable_gradient_checkpointing() takes 1 positional argument but 2 were given
```

裸库在迁移前 dev 基线和本分支均可复现。最小入口（无需启动 CS）：

```bash
python cookbook/client/swift_model/reproduce_st_dependency.py \
  --model sentence-transformers/all-MiniLM-L6-v2
# 离线时 --model 指向已有的 ST 模型目录。
```

ST 首版在服务配置中显式使用 `gradient_checkpointing: false`；这是 Swift 构造函数已有的参数，不改 Swift，也不修改其方法。该选择会增加训练显存占用。将来启用梯度检查点前，需要先修复 Swift 封装并更新固定 SHA。

服务示例使用 FP32（`mixed_precision: "no"`）、一个 on-demand 实例及两个租户槽位，模型和采样接口不混用：这里通过训练 handle 的 `forward_only` 读取原生 ST 流水线向量。

## 运行示例

在已就绪的 Twinkle 设备环境中，用 `--no-deps` 安装 `requirements-st.txt` 的固定 Swift/ST 版本，缺失的 ST 常规依赖单独补齐，然后启动：

```bash
python -m twinkle.server launch --config cookbook/client/swift_model/sentence_transformer_config.yaml
python cookbook/client/swift_model/sentence_transformer_client.py --base-url http://127.0.0.1:8000
```

服务端离线时将 YAML 的 `model_id` 改为完整 ST 模型缓存目录，保留公开路由名；客户端以 `--tokenizer` 指向相同原始模型目录。恢复必须使用同一原始 ST 模型和模块配置，adapter checkpoint 仅保存租户权重、optimizer 与训练进度。

## 验收范围

先完成 on-demand 普通模型的设备验收，再运行 ST 链路：冷启动 → 创建两个 adapter → 设置 embedding processor、loss 和 optimizer → 训练与读取向量 → 保存 adapter → 释放并重建 handle → 恢复 optimizer 和进度 → 继续训练 → 释放并复用槽位。
对照同一原始 ST 模型、模块配置和精度，比较池化、投影、归一化后的向量及参数更新；同时检查租户间 optimizer、进度、模板与权重不串用。
首版限定 LoRA，不扩展全参、SP 或 padding-free。

## 验收状态（2026-10-27）

合并 `fix/model-state-cleanup` 后，普通 Transformers 模型的真实 PPU CS 专项已重新验证双 SDK 训练、optimizer 保存恢复、Controller 重启、worker 故障恢复、创建去重、共享容量、Full 独占和资源释放；Twinkle、Tinker 恢复后继续训练的 logprob 最大误差均为 0。Redis 重启另有迁移阶段验收记录，本轮未据此宣称重新验收。
合并后的 Transformers/Megatron × Twinkle/Tinker × SFT/DPO/GRPO 核心矩阵 12/12 通过，无跳过；SFT 双 SDK 保存与 optimizer 恢复通过，两后端均确认训练资源释放。Megatron 的 adapter-disabled 参考前向默认不计算已配置的 DPO loss；普通评估、训练和显式要求计算 loss 的前向语义保持不变。
本地 ST 回归覆盖 Pooling、Dense、Normalize 流水线与租户 optimizer、进度、模板和槽位隔离。all-MiniLM-L6-v2 的 CPU 裸库训练、adapter 恢复和继续训练也已通过。

**真实 ST HTTP / PPU 验收已通过。** 覆盖双租户隔离、训练、保存、槽位复用、optimizer 与进度恢复、实例释放后的冷重建及继续训练。相同原始 MiniLM 模型、模块配置、FP32 精度及 checkpoint/RNG 起点下，CS 与裸库的向量和 adapter 参数最大误差均为 0，向量与参数均发生实际更新。
设备验收入口单独将 `idle_timeout_s` 设置为 900 秒，以覆盖裸库对照进程的冷导入和设备初始化；服务示例的 60 秒闲置配置保持不变。session 心跳不会续期训练业务闲置期限，长时间离开 CS 做本地计算需要相应配置业务闲置预算。

采样池独立 `1→2→1` 设备验收仍待完成，不能用普通训练或 ST 的通过替代。扩展用例中的 Full checkpoint IPC 热更新超时继续单独记录，不纳入本次修复。

独立设备验收入口（需要两张已确认空闲的卡；CS 使用物理卡 0，裸库对照进程使用物理卡 1）：

```bash
TWINKLE_TEST_ST_CS=1 TWINKLE_TEST_ST_MODEL=/absolute/ST/model/cache \
CUDA_VISIBLE_DEVICES=0 python -m pytest --noconftest \
  tests/server/integration/st_cs/test_http_e2e.py -q
```

采样扩缩容入口使用 `TWINKLE_TEST_REAL_POOL=1`、`TWINKLE_TEST_POOL_DEVICES=2`、`TWINKLE_TEST_MODEL_PATH` 和 `TWINKLE_TEST_ARTIFACT_DIR`，选择 `tests/server/integration/sampler/test_real_pool_e2e.py::test_real_pool_scaling`。原有 `test_real_pool_lifecycle` 继续保留 Full 热更新、LoRA 驱逐及故障恢复等扩展断言。
