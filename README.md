# esmTunel
实现ESM的模型进行微调，实现对蛋白质过敏原风险的预测。

## 模型架构
- **Backbone**: ESM-2 (facebook/esm2_t33_650M_UR50D)
- **Adapter**: LoRA (r=8, alpha=32, target modules: query, key, value)
- **Head**: 
  - Linear(1280, 512) + ReLU
  - Dropout(0.1)
  - Linear(512, 1) + Sigmoid

## 文件说明
- `model.py`: 定义模型架构，包含ESM-2骨架和LoRA适配器。
- `dataset.py`: 定义数据加载逻辑。
- `train.py`: 训练脚本，包含Early Stopping和评估指标。

## 运行环境
- torch
- transformers
- peft
- datasets
- scikit-learn
- pandas
- tqdm
