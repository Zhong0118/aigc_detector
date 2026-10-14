# sync 整合与迁移说明

## 统一基线

GitHub `sync` 是已提交的协作基线。服务器原有未提交修改只作为本次整合输入；运行中的数据、权重、缓存和日志不随源码提交。部署必须记录提交哈希和资产哈希。

## 本次整合边界

保留：

- 高分检测块的相邻合并与逐块家族溯源；
- 窗口 A0 + B0/B1/B3 融合的等权可复现基线；
- validation-only 调权协议、严格缓存身份校验和一次性 test 评估凭证；
- 文档提取、工作台展示、阈值运行时安全检查及回归测试；
- 依赖清单、模型目录和配置的脱敏修复。

不提交：

- `data/`、`models/`、`outputs/`、`logs/`、Hugging Face 缓存、checkpoint 和私有 `.env.local`；
- 服务器生成的巨大校准采样文件，除非经过独立审查并确认不含私有样本；
- 空的或无法验证来源的模型清单；
- 未通过测试的历史调权 lock、branch cache 和 test receipt。

## 阈值口径

校准文件中的 `threshold_floor` 是校准搜索下限，不是无条件的产品高阈值。运行时回退必须保持有限值且严格高于 low threshold；如果运行阈值与校准阈值不同，报告和页面不显示原阈值对应的 FPR/recall，并明确标记当前阈值尚未独立验证。AUROC 只有在其定义不依赖该 cutoff 时才保留。

## 融合产物迁移

旧窗口 A0 产物、branch cache 和 weights lock 未绑定完整的数据、模型、配方、标签顺序和窗口策略身份，不能按文件名直接复用。新调权脚本拒绝旧 cache/lock；模型加载器拒绝 hash 不符、非窗口 A0、验证窗口策略或长度不匹配的产物。训练 seed/view 的部署核验仍须对照训练记录，模型元数据自身不包含这些字段。

`scripts/fuse_window_a0_classical.py` 仅作历史预测文件比较，必须显式传入 `--acknowledge-unverified-inputs`。结果标记 `provenance_verified=false`，记录输入文件哈希，但不证明数据/模型身份；不得用于部署权重或正式 test 结论。它不会读取调权 lock，也不再混入无法逐样本对齐的 E1 汇总指标。正式调权使用 `scripts/tune_window_a0_fusion_weights.py` 的默认 `--mode validation`；锁定后使用同一输出目录显式调用 `--mode evaluate-test`。测试开始前即保留 receipt，失败中断也不自动解锁重试。不同输出目录或人为删除文件不在防篡改保证范围内。

迁移步骤：

1. 保留旧产物作为只读历史证据，不覆盖或改写；
2. 使用当前代码和已审核的数据/manifest 重新生成 validation branch cache；
3. 默认只运行 validation 调权并生成新的 lock；
4. 对新 lock 做人工审阅；
5. 需要 test 时显式运行一次 test-evaluation 模式，保留不可重复使用的 receipt；
6. 记录数据集、manifest、模型、tokenizer、经典模型、代码和运行环境哈希。

旧 test 指标不能作为新 lock 的依据。生产默认使用等权，不会自动发现或加载磁盘上的 lock。部署调优权重时，维护者必须先核对新 lock 与当前数据/模型身份，再显式设置 `config.yaml` 的 `weights`；运行时只负责权重数值有效性与模型配方校验，不代替这项部署审查。

## 凭据

`config.yaml` 只保存环境变量名；`.env.local.example` 只有占位符。历史提交曾包含凭据的项目必须在供应商后台轮换，并评估 Git 历史清理；本次变更不等同于凭据轮换，也不等同于历史重写。

## 交接检查表

- [ ] `git rev-parse HEAD` 与部署记录一致。
- [ ] `python -m pytest -q tests` 通过。
- [ ] `config.yaml` 未包含密钥字面值。
- [ ] 数据、模型、权重和校准资产有独立清单及哈希。
- [ ] 当前工作区无未提交业务代码。
- [ ] API 与 Dashboard 的实际进程路径、端口和提交哈希已记录。
- [ ] 新同学从 `sync` 建立个人功能分支，不直接在服务器工作树开发。
