from transformers import AutoModelForCausalLM

def load_model(model_name):
    """
    Load a pre-trained model from Hugging Face.
    
    Args:
        model_name (str): Name of the pre-trained model.
    
    Returns:
        model: Loaded model.
    """
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return model