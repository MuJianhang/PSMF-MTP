# Trained PSMF-MTP weights / 训练权重

The file `psmf_mtp_100epochs_seed205808.pt` is the 100-epoch, seed-205808, complete-feature PSMF-MTP fusion/classifier state dictionary. It contains 6,877,163 parameter values. The frozen sequence encoder is supplied separately in `../pre_trained_model/`.

本文件为100轮、随机种子205808、完整特征配置的 PSMF-MTP 融合与分类部分参数字典，包含6,877,163个参数值。冻结编码器另见 `../pre_trained_model/`。

- Settings and SHA-256 / 配置和校验值: [model_config.json](model_config.json)
- Individual run record / 单次训练记录: [training_record.json](training_record.json)
- SHA-256: `c70c0420428b6e7291a1a2cbe048525ab032e49acb02508f2521368fef82e1e3`

A matching local architecture definition is required to use a `state_dict` for prediction. The core implementation is retained locally; this parameter file contains no training script or custom loss source. Single-checkpoint metrics are distinct from repeated-run summaries.

预测时需要与参数匹配的本地模型定义。核心实现保留在本地，此参数文件不包含训练脚本或自定义损失源码。单个检查点的指标与多次运行统计结果有所区别。
