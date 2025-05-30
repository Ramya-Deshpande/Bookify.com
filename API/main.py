from flask import Flask, request, jsonify
from model_loader import load_model_and_tokenizer
from preprocess import preprocess_text

app = Flask(__name__)

# Simple home route for sanity check
@app.route('/')
def home():
    return "API is running"

# Load model and tokenizer once at startup
model, tokenizer = load_model_and_tokenizer()

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    text = data.get('text', '')

    if not text or not isinstance(text, str):
        return jsonify({'error': 'Invalid input, text field is required and should be a non-empty string.'}), 400

    # Preprocess input text
    sequence = preprocess_text(text, tokenizer)

    # Predict sentiment (assuming model expects padded sequence)
    pred = model.predict(sequence)
    score = float(pred[0][0])  # adjust if multiclass or softmax

    label = 'positive' if score > 0.5 else 'negative'  # example threshold

    # Logging (optional)
    app.logger.info(f"Input text: {text}")
    app.logger.info(f"Prediction: {label} with confidence {score}")

    return jsonify({'label': label, 'score': score})

if __name__ == '__main__':
    app.run(debug=True)
