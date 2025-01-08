from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    pipeline
)

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

# Step 2: Create and split the dataset
dataset = Dataset.from_dict(data)
small_dataset = dataset.train_test_split(test_size=0.2, seed=42)

# Step 3: Load the base model and tokenizer
model_name = "gpt2"  # Replace with your base model
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Assign a padding token
tokenizer.pad_token = tokenizer.eos_token

# Step 4: Tokenize the dataset
def tokenize_function(examples):
    inputs = [f"{instruction} {input_text}" for instruction, input_text in zip(examples["instruction"], examples["input"])]
    outputs = examples["output"]
    model_inputs = tokenizer(inputs, padding="max_length", truncation=True, max_length=512)
    labels = tokenizer(outputs, padding="max_length", truncation=True, max_length=512).input_ids
    model_inputs["labels"] = labels
    return model_inputs

tokenized_dataset = small_dataset.map(tokenize_function, batched=True)

# Step 5: Define training arguments
training_args = TrainingArguments(
    output_dir="./alpaca_model",
    overwrite_output_dir=True,
    evaluation_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    save_strategy="epoch",
    save_total_limit=2,
    logging_dir="./logs",
    logging_steps=10
)

# Step 6: Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer
)

# Step 7: Train the model
trainer.train()

# Step 8: Save the fine-tuned model and tokenizer
model.save_pretrained("./alpaca_model")
tokenizer.save_pretrained("./alpaca_model")

# Step 9: Generate text using the fine-tuned model
fine_tuned_model = "./alpaca_model"
alpaca_pipeline = pipeline("text-generation", model=fine_tuned_model, tokenizer=fine_tuned_model)

prompt = "What are the benefits of renewable energy?"
response = alpaca_pipeline(prompt, max_length=100, num_return_sequences=1)
print("Generated Text:")
print(response[0]["generated_text"])
