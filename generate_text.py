from transformers import AutoTokenizer, AutoModelForCausalLM

fine_tuned_model = "./alpaca_model"

# Load the model
try:
    model = AutoModelForCausalLM.from_pretrained(fine_tuned_model)
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Load the tokenizer
try:
    tokenizer = AutoTokenizer.from_pretrained(fine_tuned_model)
except Exception as e:
    print(f"Error loading tokenizer: {e}")
    exit()

# Initialize the text-generation pipeline
try:
    from transformers import pipeline
    alpaca_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)
except Exception as e:
    print(f"Error initializing pipeline: {e}")
    exit()

# Test the pipeline
prompt = "What are the benefits of renewable energy?"
response = model_pipeline(prompt, max_length=100, num_return_sequences=1, temperature=0.7, top_p=0.9)
print(response[0]["generated_text"])
