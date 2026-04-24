# 蛋白质过敏性预测系统：模型微调设计文档

## 1. 项目概述
本项目的目标是基于预训练蛋白质语言模型（pLM），通过微调（Fine-tuning）构建一个高精度的过敏原风险预测模型。系统需处理约 2,000 条蛋白质序列，并能识别复杂的结构表位（Conformational Epitopes）特征。

## 2. 技术栈
- **计算平台**: 基于 CPU/GPU 的计算环境 (推荐 Ascend 910 或 NVIDIA GPU)
- **预训练模型**: ESM-2 (facebook/esm2_t33_650M_UR50D, 650M 参数)
- **微调技术**: LoRA (Low-Rank Adaptation)
- **数据处理**: Biopython, Pandas, Scikit-learn
- **深度学习框架**: PyTorch, Hugging Face Transformers, PEFT

## 3. 模型架构设计
### 3.1 总体结构
模型采用“冻结骨架 + 动态插件 + 线性分类头”的架构：
1. **Backbone (ESM-2)**: 提取进化、理化及隐式结构特征。
2. **LoRA Adapter**: 在 Transformer 的 Q/K/V 矩阵中插入低秩分解矩阵。秩 (r) = 8，Alpha = 32。
3. **Classification Head**:
   - 输入: ESM-2 输出的 [CLS] token (1280 维)。
   - Layer 1: Linear(1280, 512) + ReLU + Dropout(0.1)。
   - Layer 2: Linear(512, 1) + Sigmoid。

## 4. 微调策略
### 4.1 LoRA 配置
- **秩 (r)**: 8
- **Alpha**: 32
- **目标层 (Target Modules)**: query, key, value
- **Dropout**: 0.1
- **可训练参数占比**: 约 0.3% (针对 650M 模型)

### 4.2 训练超参数
- **学习率 (LR)**: 1e-4
- **Batch Size**: 16 (根据显存调整)
- **训练轮数 (Epochs)**: 10
- **优化器**: AdamW (Weight Decay = 0.01)
- **损失函数**: BCELoss (二分类交叉熵)
- **早停策略 (Early Stopping)**: 监控验证集损失，耐心值 (Patience) = 3

## 5. 评估指标
- **Accuracy**: 总体预测准确率
- **Precision / Recall / F1-score**: 针对过敏原类别的识别性能
- **AUC-ROC**: 衡量模型在不同阈值下的分类能力

## 6. 实施细节
### 6.1 显存优化
- **Float16 训练**: 减少权重和激活值的内存占用。
- **Gradient Checkpointing**: 以时间换空间，进一步降低训练时的显存需求。
- **Parameter-Efficient**: 仅更新 LoRA 权重，极大降低显存和存储消耗。

### 6.2 代码结构
- `model.py`: 模型定义与 LoRA 配置。
- `dataset.py`: 蛋白质序列 tokenization 与加载。
- `train.py`: 训练、验证与评估主逻辑。
- `scripts/preprocess_data.py`: 数据清洗与去冗余。
- `scripts/plot_results.py`: 训练指标可视化。
