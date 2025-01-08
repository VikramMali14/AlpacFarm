from datasets import Dataset

# Step 1: Define the dataset
data = {
    "instruction": [
        "What are the benefits of renewable energy?",
        "Explain the greenhouse effect.",
        "Define machine learning.",
        "What is the significance of data preprocessing?"
    ],
    "input": [
        "Explain briefly.",
        "Provide details.",
        "Give a simple definition.",
        "Why is it important?"
    ],
    "output": [
        "Renewable energy reduces pollution and conserves resources.",
        "The greenhouse effect traps heat in Earth's atmosphere due to gases like CO2.",
        "Machine learning is a field of AI that enables systems to learn from data.",
        "Data preprocessing improves model accuracy and efficiency."
    ]
}

# Step 2: Create the Dataset object
dataset = Dataset.from_dict(data)

# Step 3: Split the dataset into train and test sets
try:
    small_dataset = dataset.train_test_split(test_size=0.2, seed=42)
    print("Dataset split successfully!")
except ValueError as e:
    print(f"Error splitting dataset: {e}")
    small_dataset = {"train": dataset, "test": dataset}  # Use the full dataset if split fails

# Step 4: Display the split dataset
print("Training Dataset:")
print(small_dataset["train"])

print("Testing Dataset:")
print(small_dataset["test"])

# Step 5: Save the smaller dataset
train_path = "train_dataset.json"
test_path = "test_dataset.json"

small_dataset["train"].to_json(train_path, orient="records", lines=True)
small_dataset["test"].to_json(test_path, orient="records", lines=True)

print(f"Training dataset saved to {train_path}")
print(f"Testing dataset saved to {test_path}")
