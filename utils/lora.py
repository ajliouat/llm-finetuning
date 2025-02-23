from peft import LoraConfig, get_peft_model

def apply_lora(model, lora_config):
    """
    Apply LoRA to the model.
    
    Args:
        model: Pre-trained model.
        lora_config (dict): Configuration for LoRA.
    
    Returns:
        model: Model with LoRA applied.
    """
    config = LoraConfig(
        r=lora_config['r'],
        lora_alpha=lora_config['lora_alpha'],
        target_modules=lora_config['target_modules'],
        lora_dropout=lora_config['lora_dropout'],
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, config)
    return model