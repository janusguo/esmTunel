import torch
import torch.nn as nn
from transformers import AutoModel
from peft import get_peft_model, LoraConfig, TaskType

class ESMAllergenicityModel(nn.Module):
    def __init__(self, model_name="facebook/esm2_t33_650M_UR50D", lora_r=8, lora_alpha=32, attn_implementation=None):
        super(ESMAllergenicityModel, self).__init__()
        # Load ESM-2 backbone with float16 to save memory
        print(f"Loading backbone model: {model_name}")
        kwargs = {"torch_dtype": torch.float16}
        if attn_implementation:
            kwargs["attn_implementation"] = attn_implementation
        self.esm = AutoModel.from_pretrained(model_name, **kwargs)
        
        # Enable gradient checkpointing to save memory during fine-tuning
        self.esm.gradient_checkpointing_enable()
        
        # LoRA Configuration
        peft_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            inference_mode=False,
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=["query", "key", "value"],
            lora_dropout=0.1
        )
        
        # Apply LoRA to ESM-2
        self.esm = get_peft_model(self.esm, peft_config)
        self.esm.print_trainable_parameters()
        
        # Classification Head
        # ESM-2 650M has hidden size 1280
        hidden_size = self.esm.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 1)
            # Sigmoid removed to use BCEWithLogitsLoss for stability
        )
        
        # Ensure classifier matches backbone dtype
        self.classifier.to(self.esm.dtype)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.esm(input_ids=input_ids, attention_mask=attention_mask)
        # Use [CLS] token representation (the first token)
        # For ESM, the first token is <cls>
        cls_output = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_output)
        return logits
