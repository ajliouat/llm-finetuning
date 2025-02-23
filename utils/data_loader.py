from datasets import load_dataset

def load_data(data_path):
    """
    Load and preprocess the dataset from a CSV file.
    
    Args:
        data_path (str): Path to the dataset file.
    
    Returns:
        train_dataset: Training dataset.
        eval_dataset: Evaluation dataset.
    """
    dataset = load_dataset('csv', data_files=data_path)
    train_dataset = dataset['train']
    eval_dataset = dataset['validation']
    return train_dataset, eval_dataset