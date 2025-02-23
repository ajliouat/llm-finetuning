from transformers import GPT2LMHeadModel
from peft import LoraConfig, get_peft_model

class LoRAModel(GPT2LMHeadModel):
    """
    Custom model class with LoRA applied.
    """
    def __init__(self, config):
        super().__init__(config)
        self.lora_config = LoraConfig(
            r=8,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM"
        )
        # Apply LoRA to the model
        self.model = get_peft_model(self, self.lora_config)