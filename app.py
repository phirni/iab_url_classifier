from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import requests
from bs4 import BeautifulSoup
import re
import os
import json

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend

# Load model and tokenizer
MODEL_PATH = "./model"
tokenizer = None
model = None
LABEL_NAMES = []

# Load label names from enhanced training report
try:
    with open(os.path.join(MODEL_PATH, "enhanced_training_report.json"), "r") as f:
        report = json.load(f)
        LABEL_NAMES = report["model_info"]["label_names"]
    print(f"Loaded {len(LABEL_NAMES)} label names from training report")
except Exception as e:
    print(f"Error loading label names from enhanced_training_report.json: {e}")

def load_model():
    global tokenizer, model
    try:
        # Use base DistilBERT tokenizer since custom tokenizer files are missing
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
        model.eval()
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")

def extract_text_from_url(url):
    """Extract text content from URL"""
    try:
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url

        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }

        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, 'html.parser')

        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()

        # Extract text from title and meta description
        title = soup.find('title')
        title_text = title.get_text() if title else ""

        meta_desc = soup.find('meta', attrs={'name': 'description'})
        desc_text = meta_desc.get('content', '') if meta_desc else ""

        # Extract main content
        body_text = soup.get_text()

        # Clean and combine text
        text = f"{title_text} {desc_text} {body_text}"
        text = re.sub(r'\s+', ' ', text).strip()

        return text[:2000]  # Limit text length

    except Exception as e:
        print(f"Error extracting text from URL: {e}")
        return ""

def predict_iab_categories(text, threshold=0.3, top_k=5):
    """Predict IAB categories for given text"""
    try:
        # Tokenize input
        inputs = tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="pt"
        )

        # Make prediction
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.sigmoid(outputs.logits)

        # Convert to numpy
        predictions = predictions.numpy()[0]

        # Get top predictions above threshold
        top_indices = np.where(predictions > threshold)[0]

        # Sort by confidence
        sorted_indices = top_indices[np.argsort(predictions[top_indices])[::-1]]

        # Limit to top_k
        sorted_indices = sorted_indices[:top_k]

        # Create results
        results = []
        for idx in sorted_indices:
            label = LABEL_NAMES[idx] if idx < len(LABEL_NAMES) else f"LABEL_{idx}"
            results.append({
                'category_id': int(idx),
                'category_name': label,
                'confidence': float(predictions[idx])
            })

        return results

    except Exception as e:
        print(f"Error in prediction: {e}")
        return []

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "model_loaded": model is not None})

@app.route('/predict_url', methods=['POST'])
def predict_url():
    """Main prediction endpoint for URLs"""
    try:
        data = request.get_json()
        url = data.get('url', '').strip()
        threshold = float(data.get('threshold', 0.3))
        top_k = int(data.get('top_k', 5))

        if not url:
            return jsonify({"error": "URL is required"}), 400

        # Extract text from URL
        text = extract_text_from_url(url)

        if not text:
            return jsonify({"error": "Could not extract text from URL"}), 400

        # Predict IAB categories
        predictions = predict_iab_categories(text, threshold=threshold, top_k=top_k)

        return jsonify({
            "url": url,
            "extracted_text_preview": text[:200] + "..." if len(text) > 200 else text,
            "predictions": predictions,
            "total_categories": len(predictions)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/predict_text', methods=['POST'])
def predict_text():
    """Predict from direct text input"""
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        threshold = float(data.get('threshold', 0.3))
        top_k = int(data.get('top_k', 5))

        if not text:
            return jsonify({"error": "Text is required"}), 400

        # Predict IAB categories
        predictions = predict_iab_categories(text, threshold=threshold, top_k=top_k)

        return jsonify({
            "text": text,
            "predictions": predictions,
            "total_categories": len(predictions)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    load_model()
    port = int(os.environ.get('PORT', 8000))
    app.run(debug=False, host='0.0.0.0', port=port)