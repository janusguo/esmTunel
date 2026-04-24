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
- `train.py`: 训练脚本，支持命令行参数，包含Early Stopping和评估指标。
- `scripts/preprocess_data.py`: 数据预处理脚本。
- `scripts/plot_results.py`: 训练结果可视化脚本。

## 运行环境
- torch
- transformers
- peft
- datasets
- scikit-learn
- pandas
- tqdm
- matplotlib
- biopython

## 使用方法
1. 数据预处理:
   ```bash
   python scripts/preprocess_data.py
   ```
2. 模型训练:
   ```bash
   python train.py --model_name facebook/esm2_t33_650M_UR50D --batch_size 16 --epochs 10
   ```
3. 结果可视化:
   ```bash
   python scripts/plot_results.py
   ```
