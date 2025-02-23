import argparse
import yaml
from utils.data_loader import load_data
from utils.model_utils import load_model
from utils.lora import apply_lora
from transformers import Trainer, TrainingArguments

def main(config):
    # Load data
    train_dataset, eval_dataset = load_data(config['data_path'])

    # Load model
    model = load_model(config['model_name'])

    # Apply LoRA
    model = apply_lora(model, config['lora_config'])

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=config['output_dir'],
        per_device_train_batch_size=config['batch_size'],
        per_device_eval_batch_size=config['batch_size'],
        num_train_epochs=config['epochs'],
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir=config['logging_dir'],
        logging_steps=10,
        save_total_limit=2,
        fp16=config['fp16'],
        gradient_checkpointing=config['gradient_checkpointing'],
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    # Train the model
    trainer.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    main(config)