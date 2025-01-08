from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model

# Step 1: Load the dataset
dataset = load_dataset("json", data_files="./alpaca_data.json")

# Step 2: Access the 'train' split and shuffle it
train_dataset = dataset['train'].shuffle(seed=42)

# Step 3: Select the first 1000 examples for smaller dataset
small_dataset = train_dataset.select(range(1000))

# Print the dataset details
print(small_dataset)

# Step 4: Load the pre-trained model and tokenizer
model_name = "EleutherAI/gpt-neo-1.3B"  # Example model, you can change it to your desired one
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Set the padding token if not already set
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))


# Step 5: Apply LoRA for efficient fine-tuning
lora_config = LoraConfig(
    task_type="CAUSAL_LM", 
    inference_mode=False, 
    r=4,  # Lower rank
    lora_alpha=16, 
    lora_dropout=0.1
)

# Apply LoRA to the model
model = get_peft_model(model, lora_config)

# Step 6: Tokenize the dataset
def tokenize_function(examples):
    # Combine 'instruction' and 'input' as context for the model, and use 'output' as the target
    inputs = [instr + inp for instr, inp in zip(examples['instruction'], examples['input'])]
    targets = examples['output']
    
    # Tokenize both inputs and targets with consistent max_length
    model_inputs = tokenizer(inputs, padding="max_length", truncation=True, max_length=512)
    
    # Tokenize targets separately, then match the padding length
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, padding="max_length", truncation=True, max_length=512)

    # Set labels for the causal language modeling task
    model_inputs["labels"] = labels["input_ids"]

    # Replace padding token ID in labels with -100 to ignore them during loss computation
    model_inputs["labels"] = [
        [(label if label != tokenizer.pad_token_id else -100) for label in label_list]
        for label_list in model_inputs["labels"]
    ]

    return model_inputs


# Apply the tokenizer to the dataset
small_dataset = small_dataset.map(tokenize_function, batched=True)

# Step 7: Define the Training Arguments
training_args = TrainingArguments(
    output_dir="./alpaca_model",  # Directory to save the model
    per_device_train_batch_size=1,  # Smaller batch size for lower-spec machines
    gradient_accumulation_steps=16,  # Accumulate gradients for larger batch size
    num_train_epochs=1,  # Start with 1 epoch for testing
    learning_rate=2e-5,
    fp16=False,  # Disable FP16 if not supported by your machine
    save_steps=100,  # Save model every 100 steps
    save_total_limit=2,  # Keep only 2 saved models at most
    logging_dir="./logs",  # Directory for logging
)

# Step 8: Define the Trainer and start training
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_dataset,  # Use the preprocessed small dataset
)

# Start the training process
trainer.train()
