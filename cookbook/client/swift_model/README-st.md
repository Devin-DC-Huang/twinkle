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

先在服务端安装 `requirements-st.txt`，启动：

```bash
python -m twinkle.server launch --config cookbook/client/swift_model/sentence_transformer_config.yaml
python cookbook/client/swift_model/sentence_transformer_client.py --base-url http://127.0.0.1:8000
```

服务端离线时将 YAML 的 `model_id` 改为完整 ST 模型缓存目录，保留公开路由名；客户端以 `--tokenizer` 指向相同原始模型目录。恢复必须使用同一原始 ST 模型和模块配置，adapter checkpoint 仅保存租户权重、optimizer 与训练进度。

## 验收范围

先完成 on-demand 普通模型的设备验收，再运行 ST 链路：冷启动 → 创建两个 adapter → 设置 embedding processor、loss 和 optimizer → 训练与读取向量 → 保存 adapter → 释放并重建 handle → 恢复 optimizer 和进度 → 继续训练 → 释放并复用槽位。
对照同一原始 ST 模型、模块配置和精度，比较池化、投影、归一化后的向量及参数更新；同时检查租户间 optimizer、进度、模板与权重不串用。
首版限定 LoRA，不扩展全参、SP 或 padding-free。
