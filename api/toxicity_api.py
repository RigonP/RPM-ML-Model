# toxicity_api.py
from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np
from googletrans import Translator  # Shto këtë
import os

app = Flask(__name__)
CORS(app)

# Inicializo përkthyesin
translator = Translator()

print("Po ngarkoj modelin...")

# Ngarko modelin dhe tokenizer-in
try:
    model = load_model('toxicity_model.h5')
    print("✅ Modeli u ngarkua")
    
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    print("✅ Tokenizer-i u ngarkua")
    
    with open('config.pickle', 'rb') as handle:
        config = pickle.load(handle)
        max_len = config['max_len']
    print(f"✅ Konfigurimi u ngarkua (max_len={max_len})")
    
except Exception as e:
    print(f"❌ Gabim gjatë ngarkimit: {e}")
    raise

def detect_and_translate(text):
    """
    Detekton gjuhën dhe përkthyen në anglisht nëse është e nevojshme
    """
    try:
        # Detekto gjuhën
        detection = translator.detect(text)
        detected_lang = detection.lang
        confidence = detection.confidence
        
        print(f"Gjuha e detektuar: {detected_lang} (besueshmëri: {confidence})")
        
        # Nëse nuk është anglisht, përkthe
        if detected_lang != 'en':
            translated = translator.translate(text, src=detected_lang, dest='en')
            english_text = translated.text
            print(f"Përkthyer: {text[:50]}... -> {english_text[:50]}...")
            return english_text, detected_lang
        
        return text, 'en'
    
    except Exception as e:
        print(f"⚠️ Gabim në përkthim: {e}. Po përdor tekstin origjinal.")
        # Nëse ka problem, përdor tekstin origjinal
        return text, 'unknown'

@app.route('/health', methods=['GET'])
def health_check():
    """Kontrollo nëse API është aktiv"""
    return jsonify({
        'status': 'healthy',
        'message': 'Toxicity Detection API është aktiv'
    })

@app.route('/predict', methods=['POST'])
def predict_toxicity():
    """Parashiko toksicitetin e një teksti"""
    try:
        # Merr tekstin nga request
        data = request.json
        original_text = data.get('text', '')
        
        if not original_text:
            return jsonify({'error': 'Nuk u dërgua asnjë tekst'}), 400
        
        print(f"Po analizoj: {original_text[:50]}...")
        
        # Përkthe nëse është e nevojshme
        text_to_analyze, detected_language = detect_and_translate(original_text)
        
        # Përgatit tekstin
        sequence = tokenizer.texts_to_sequences([text_to_analyze])
        padded = pad_sequences(sequence, maxlen=max_len, padding='post')
        
        # Bëj parashikimin
        prediction = model.predict(padded, verbose=0)[0][0]
        toxicity_score = float(prediction * 100)
        
        # Vendos pragun në 80%
        is_toxic = toxicity_score >= 80
        
        result = {
            'toxicity_score': round(toxicity_score, 2),
            'is_toxic': is_toxic,
            'original_text': original_text,
            'analyzed_text': text_to_analyze if detected_language != 'en' else None,
            'detected_language': detected_language,
            'message': 'Teksti është toksik' if is_toxic else 'Teksti është i pranueshëm'
        }
        
        print(f"Rezultati: {toxicity_score:.2f}% - {'TOKSIK' if is_toxic else 'OK'}")
        
        return jsonify(result)
    
    except Exception as e:
        print(f"❌ Gabim: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/batch-predict', methods=['POST'])
def batch_predict():
    """Analizo shumë tekste njëkohësisht"""
    try:
        data = request.json
        texts = data.get('texts', [])
        
        if not texts or not isinstance(texts, list):
            return jsonify({'error': 'Duhet të dërgoni një listë tekstesh'}), 400
        
        results = []
        for original_text in texts:
            # Përkthe nëse është e nevojshme
            text_to_analyze, detected_language = detect_and_translate(original_text)
            
            sequence = tokenizer.texts_to_sequences([text_to_analyze])
            padded = pad_sequences(sequence, maxlen=max_len, padding='post')
            prediction = model.predict(padded, verbose=0)[0][0]
            toxicity_score = float(prediction * 100)
            
            results.append({
                'original_text': original_text,
                'analyzed_text': text_to_analyze if detected_language != 'en' else None,
                'detected_language': detected_language,
                'toxicity_score': round(toxicity_score, 2),
                'is_toxic': toxicity_score >= 90
            })
        
        return jsonify({'results': results})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("\n" + "="*50)
    print("🚀 Toxicity Detection API është duke u nisur...")
    print("🌍 Suporton të gjitha gjuhët (me përkthim automatik)")
    print("="*50)
    print("📍 URL: http://localhost:5000")
    print("📍 Health Check: http://localhost:5000/health")
    print("📍 Predict: POST http://localhost:5000/predict")
    print("="*50 + "\n")
    
    app.run(host='0.0.0.0', port=5000, debug=True)