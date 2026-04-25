# Kaggle Fine-tuning Guide for Protein Allergenicity Prediction

本指南提供了在 Kaggle GPU 环境（如 Tesla T4 或 P100）上运行 ESM-2 微调和推理流水线的详细步骤。

## 1. Kaggle 环境设置

1. 创建一个新的 Kaggle Notebook。
2. 在右侧的 **Settings** 菜单中，启用 **GPU Acceleration** (T4 或 P100)。
3. 确保 **Internet access** 已打开，以便下载预训练模型。

## 2. 安装依赖项

在第一个单元格中运行以下命令：

```python
!pip install -q transformers peft pandas scikit-learn matplotlib seaborn tqdm
```

## 3. 数据准备

将 `train.csv`, `val.csv`, 和 `test.csv` 上传为 Kaggle Dataset。上传后，它们将位于 `/kaggle/input/<your-dataset-name>/` 目录下。

## 4. 核心代码实现

将以下代码块复制到 Kaggle Notebook 的不同单元格中。

### 4.1 数据集处理

```python
import torch
from torch.utils.data import Dataset
import pandas as pd
from transformers import AutoTokenizer

class ProteinDataset(Dataset):
    def __init__(self, csv_path, tokenizer_name="facebook/esm2_t33_650M_UR50D", max_length=512):
        self.df = pd.read_csv(csv_path)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        sequence = self.df.iloc[idx]['sequence']
        label = self.df.iloc[idx]['label']
        encoding = self.tokenizer(
            sequence, truncation=True, padding='max_length',
            max_length=self.max_length, return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.float)
        }
```

### 4.2 模型架构 (ESM-2 + LoRA)

```python
import torch.nn as nn
from transformers import AutoModel
from peft import get_peft_model, LoraConfig, TaskType

class ESMAllergenicityModel(nn.Module):
    def __init__(self, model_name="facebook/esm2_t33_650M_UR50D", lora_r=8, lora_alpha=32):
        super(ESMAllergenicityModel, self).__init__()
        # 使用 float16 以节省内存
        self.esm = AutoModel.from_pretrained(model_name, torch_dtype=torch.float16)
        # 启用梯度检查点以进一步降低显存占用
        self.esm.gradient_checkpointing_enable()
        
        peft_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            inference_mode=False, r=lora_r, lora_alpha=lora_alpha,
            target_modules=["query", "key", "value"], lora_dropout=0.1
        )
        self.esm = get_peft_model(self.esm, peft_config)
        
        hidden_size = self.esm.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 1)
        )
        self.classifier.to(self.esm.dtype)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.esm(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_output)
        return logits
```

### 4.3 训练与评估

建议直接使用本项目中的 `train.py` 逻辑。在 Kaggle 中，你可以通过以下方式调用：

```python
# 假设你已经将 GitHub 仓库克隆到本地或将脚本复制到了当前工作目录
# !python train.py --model_name facebook/esm2_t33_650M_UR50D --batch_size 16 --epochs 10
```

## 5. 模型解释性 (Conformational Epitopes)

为了识别蛋白质中的结构表位，可以使用 `scripts/interpret_model.py` 生成注意力热图和显著性图。

```python
# 在 Notebook 中调用解释性脚本的示例
# !python scripts/interpret_model.py --model_path best_model.pt --sequence <YOUR_SEQUENCE>
```

## 6. Kaggle 优化技巧

- **显存优化**: 必须使用 `torch.float16` 和 LoRA。对于 650M 模型，Batch Size 建议设为 16。
- **持久化**: 训练好的模型应保存到 `/kaggle/working/best_model.pt`。
- **早期停止**: 监控验证集损失，防止在约 2,000 条数据的小样本集上过拟合。
