import os
import cv2
import numpy as np
import easyocr
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from PIL import Image
from textblob import TextBlob

app = Flask(__name__, static_folder="../frontend", static_url_path="")
CORS(app)

print("Loading Neural Network...")
reader = easyocr.Reader(['en'], gpu=False)
print("System Ready.")

def intelligent_line_removal(img_cv):
    """
    BEST METHOD for Lined Paper:
    Uses the Red Channel to naturally fade out red/pink notebook lines.
    """

    b, g, r = cv2.split(img_cv)

    combo = cv2.addWeighted(r, 0.85, g, 0.15, 0)

    thresh = cv2.adaptiveThreshold(combo, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY, 15, 10)
    
    clean = cv2.fastNlMeansDenoising(thresh, None, 10, 7, 21)
    return clean

def calculate_stability(img_cv):
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    _, bin_img = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours: return 0, "No Input", "N/A"

    y_points = []
    heights = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if h > 10 and w > 5:
            y_points.append(y + h/2)
            heights.append(h)

    if not y_points: return 0, "Error", "N/A"

    wobble = np.std(y_points)
    size_var = np.std(heights)
    
    score = max(0, min(100, 100 - (wobble * 0.6 + size_var * 0.4)))
    
    if score > 75: label, plan = "Steady / Neuro-Typical", "None needed."
    elif score > 50: label, plan = "Mild Irregularity", "Tracing exercises."
    else: label, plan = "Possible Dysgraphia", "Consult OT."
    
    return round(score, 1), label, plan

@app.route("/")
def serve_home():
    return send_from_directory(app.static_folder, "index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    if "file" not in request.files:
        return jsonify({"error": "No file"}), 400
    file = request.files["file"]

    try:
        img = Image.open(file.stream).convert('RGB')
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        
        clean_img = intelligent_line_removal(img_cv)

        results = reader.readtext(clean_img, detail=1, paragraph=False)
        
        extracted_words = []
        total_confidence = 0
        word_count = 0

        for (bbox, text, conf) in results:
            extracted_words.append(text)
            total_confidence += conf
            word_count += 1
        
        avg_conf = (total_confidence / word_count) if word_count > 0 else 0
        
        
        if avg_conf < 0.4:
            final_text = "⚠️ Handwriting too unstable for accurate transcription."
            status = "Low Confidence"
        else:
            raw_text = " ".join(extracted_words)
            final_text = str(TextBlob(raw_text).correct())
            status = "High Confidence"

        score, label, plan = calculate_stability(img_cv)

        return jsonify({
            "ocr_text": final_text,
            "ocr_status": status,
            "report": {
                "score": score,
                "diagnosis": label,
                "plan": plan
            }
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)