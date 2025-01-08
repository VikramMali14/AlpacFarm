from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS for enabling cross-origin requests
from transformers import pipeline, AutoTokenizer

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize the tokenizer and model pipeline
tokenizer = AutoTokenizer.from_pretrained("gpt2")  # Replace with your tokenizer if needed
model_pipeline = pipeline("text-generation", model="./alpaca_model", tokenizer=tokenizer)

@app.route('/generate', methods=['POST'])
def generate():
    try:
        data = request.json
        prompt = data.get("prompt", "")
        
        if not prompt:
            return jsonify({"error": "No prompt provided"}), 400
        
        response = model_pipeline(prompt, max_length=100, num_return_sequences=1)
        
        return jsonify({"generated_text": response[0]["generated_text"]})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(port=5000)
