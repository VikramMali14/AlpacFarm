from datasets import load_dataset

# Load the dataset
dataset = load_dataset("json", data_files="./alpaca_data.json")

# Shuffle and create a smaller dataset for testing
small_dataset = dataset.shuffle(seed=42).select(range(1000))

print(small_dataset)
