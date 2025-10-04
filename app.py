from flask import Flask, request, jsonify, render_template
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Initialize the Flask application
app = Flask(__name__)

# --- ML Model Loading ---
# Load the fine-tuned model and tokenizer from the local directory
# This is done once when the server starts to save time on each prediction
MODEL_PATH = "finbert-financial-sentiment-model"
print("Loading model...")
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
print("Model loaded successfully!")

# Define the sentiment labels
label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}

def predict_sentiment(text):
    """
    Predicts the sentiment of a given text using the loaded FinBERT model.
    """
    # Tokenize the input text
    inputs = tokenizer(text, padding=True, truncation=True, max_length=128, return_tensors="pt")

    # Perform prediction
    with torch.no_grad():
        outputs = model(**inputs)
        # Get the predicted class index by finding the max logit
        prediction_index = torch.argmax(outputs.logits, dim=1).item()

    # Map the index to its corresponding label
    sentiment = label_map[prediction_index]
    return sentiment

# --- API Routes ---
@app.route('/')
def home():
    """
    Renders the main page (index.html).
    """
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Receives text from the frontend, predicts sentiment, and returns it as JSON.
    """
    try:
        # Get the text from the JSON request body
        data = request.get_json()
        text_to_analyze = data['text']

        if not text_to_analyze.strip():
            return jsonify({'error': 'Text cannot be empty.'}), 400

        # Get the prediction from our model
        sentiment = predict_sentiment(text_to_analyze)

        # Return the result in JSON format
        return jsonify({'sentiment': sentiment})

    except Exception as e:
        print(f"An error occurred: {e}")
        return jsonify({'error': 'Failed to process the request.'}), 500

if __name__ == '__main__':
    # This runs the web server
    app.run(debug=True, port=5000)