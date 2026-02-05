# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import joblib
# import os

# app = Flask(__name__)
# CORS(app)  # allow frontend (localhost or any) to call this API

# # ✅ Load trained pipeline safely
# MODEL_PATH = os.path.join(os.path.dirname(__file__), "../model/review_pipeline.pkl")
# pipeline = joblib.load(MODEL_PATH)

# # ✅ Root route (so you don’t get 404 on visiting http://127.0.0.1:5000/)
# @app.route("/", methods=["GET"])
# def home():
#     return jsonify({"message": "Fake Review Detection API is running 🚀"})

# # ✅ Prediction endpoint
# @app.route("/predict", methods=["POST"])
# def predict():
#     data = request.get_json(silent=True) or {}
#     review = data.get("review", "").strip()

#     if not review:
#         return jsonify({"error": "No review text provided"}), 400

#     pred = pipeline.predict([review])[0]
#     proba = pipeline.predict_proba([review])[0].tolist()  # [fake, real]

#     return jsonify({
#         "prediction": int(pred),   # 0=fake, 1=real
#         "result": "Real Review ✅" if pred == 1 else "Fake Review ❌",
#         "probability": {"fake": proba[0], "real": proba[1]}
#     })

# if __name__ == "__main__":
#     app.run(debug=True, port=5000)









# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import joblib
# import os

# # Hugging Face imports for BERT
# from transformers import BertTokenizer, BertForSequenceClassification
# import torch
# import shap

# app = Flask(__name__)
# CORS(app)

# # ============================
# # CONFIG: choose model type
# # ============================
# MODEL_TYPE = "logreg"   # options: "logreg", "bert"

# # ============================
# # Load model
# # ============================
# if MODEL_TYPE == "logreg":
#     MODEL_PATH = os.path.join(os.path.dirname(__file__), "../model/review_pipeline.pkl")
#     SHAP_PATH = os.path.join(os.path.dirname(__file__), "../model/shap_explainer.pkl")
#     pipeline = joblib.load(MODEL_PATH)
#     explainer = joblib.load(SHAP_PATH)

# elif MODEL_TYPE == "bert":
#     MODEL_PATH = os.path.join(os.path.dirname(__file__), "../model/bert_model")
#     tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
#     model = BertForSequenceClassification.from_pretrained(MODEL_PATH)

# # ============================
# # Routes
# # ============================
# @app.route("/", methods=["GET"])
# def home():
#     return jsonify({"message": "Fake Review Detection API is running 🚀"})


# @app.route("/predict", methods=["POST"])
# def predict():
#     data = request.get_json(silent=True) or {}
#     review = data.get("review", "").strip()

#     if not review:
#         return jsonify({"error": "No review text provided"}), 400

#     if MODEL_TYPE == "logreg":
#         pred = pipeline.predict([review])[0]
#         proba = pipeline.predict_proba([review])[0].tolist()

#     elif MODEL_TYPE == "bert":
#         inputs = tokenizer(review, return_tensors="pt", truncation=True, padding=True, max_length=128)
#         with torch.no_grad():
#             outputs = model(**inputs)
#             probs = torch.softmax(outputs.logits, dim=1).tolist()[0]
#         pred = int(torch.argmax(torch.tensor(probs)))
#         proba = probs

#     return jsonify({
#         "prediction": int(pred),
#         "result": "Real Review ✅" if pred == 1 else "Fake Review ❌",
#         "probability": {"fake": proba[0], "real": proba[1]}
#     })


# @app.route("/explain", methods=["POST"])
# def explain():
#     """Return SHAP word importance (only for logistic regression)."""
#     if MODEL_TYPE != "logreg":
#         return jsonify({"error": "Explanation available only for Logistic Regression model"}), 400

#     data = request.get_json(silent=True) or {}
#     review = data.get("review", "").strip()
#     if not review:
#         return jsonify({"error": "No review text provided"}), 400

#     # Compute SHAP values
#     shap_values = explainer(pipeline.named_steps["tfidf"].transform([review]))
#     feature_names = pipeline.named_steps["tfidf"].get_feature_names_out()

#     # Sort by absolute importance and take top 10
#     contributions = sorted(
#         list(zip(feature_names, shap_values.values[0])),
#         key=lambda x: abs(x[1]),
#         reverse=True
#     )[:10]

#     # Reformat for frontend readability
#     results = [
#         {
#             "word": word,
#             "weight": float(weight),
#             "impact": "fake" if weight > 0 else "real"
#         }
#         for word, weight in contributions
#     ]

#     return jsonify({
#         "review": review,
#         "top_words": results
#     })


# if __name__ == "__main__":
#     app.run(debug=True, port=5000)


















from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import os
import numpy as np

# Hugging Face imports for BERT
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import shap

app = Flask(__name__)
CORS(app)

# ============================
# CONFIG: choose model type
# ============================
MODEL_TYPE = "logreg"   # options: "logreg", "bert"

# ============================
# Load model
# ============================
if MODEL_TYPE == "logreg":
    MODEL_PATH = os.path.join(os.path.dirname(__file__), "../model/review_pipeline.pkl")
    SHAP_PATH = os.path.join(os.path.dirname(__file__), "../model/shap_explainer.pkl")
    
    try:
        pipeline = joblib.load(MODEL_PATH)
        explainer = joblib.load(SHAP_PATH)
        print("✅ Logistic Regression model and SHAP explainer loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        pipeline = None
        explainer = None

elif MODEL_TYPE == "bert":
    MODEL_PATH = os.path.join(os.path.dirname(__file__), "../model/bert_model")
    try:
        tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
        model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
        print("✅ BERT model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading BERT model: {e}")

# ============================
# Routes
# ============================
@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Fake Review Detection API is running"})


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(silent=True) or {}
        review = data.get("review", "").strip()

        if not review:
            return jsonify({"error": "No review text provided"}), 400

        if MODEL_TYPE == "logreg":
            if pipeline is None:
                return jsonify({"error": "Model not loaded"}), 500
                
            pred = pipeline.predict([review])[0]
            proba = pipeline.predict_proba([review])[0].tolist()

        elif MODEL_TYPE == "bert":
            inputs = tokenizer(review, return_tensors="pt", truncation=True, padding=True, max_length=128)
            with torch.no_grad():
                outputs = model(**inputs)
                probs = torch.softmax(outputs.logits, dim=1).tolist()[0]
            pred = int(torch.argmax(torch.tensor(probs)))
            proba = probs

        return jsonify({
            "prediction": int(pred),
            "result": "Real Review ✅" if pred == 1 else "Fake Review ❌",
            "probability": {"fake": proba[0], "real": proba[1]},
            "review": review  # Include original review for explanation
        })

    except Exception as e:
        print(f"Error in predict: {e}")
        return jsonify({"error": "Prediction failed"}), 500


@app.route("/explain", methods=["POST"])
def explain():
    """Return SHAP word importance (only for logistic regression)."""
    try:
        if MODEL_TYPE != "logreg":
            return jsonify({"error": "Explanation available only for Logistic Regression model"}), 400

        if pipeline is None or explainer is None:
            return jsonify({"error": "Model or explainer not loaded"}), 500

        data = request.get_json(silent=True) or {}
        review = data.get("review", "").strip()
        
        if not review:
            return jsonify({"error": "No review text provided"}), 400

        print(f"🔍 Explaining review: {review[:100]}...")

        # Transform the review using the pipeline's TF-IDF vectorizer
        tfidf_matrix = pipeline.named_steps["tfidf"].transform([review])
        
        # Get SHAP values
        shap_values = explainer(tfidf_matrix)
        
        # Handle different SHAP return types
        if hasattr(shap_values, 'values'):
            # For newer SHAP versions
            values_array = shap_values.values[0]
        else:
            # For older SHAP versions
            values_array = shap_values[0]

        # Get feature names
        feature_names = pipeline.named_steps["tfidf"].get_feature_names_out()

        # Create word-importance pairs
        word_importance_pairs = []
        for i, (word, importance) in enumerate(zip(feature_names, values_array)):
            if abs(importance) > 1e-6:  # Filter out very small values
                word_importance_pairs.append((word, float(importance)))

        # Sort by absolute importance and take top 15
        contributions = sorted(
            word_importance_pairs,
            key=lambda x: abs(x[1]),
            reverse=True
        )[:15]

        print(f"📊 Top contributions: {contributions[:5]}")

        # Format for frontend
        # IMPORTANT: Positive SHAP values push towards class 1 (Real), negative towards class 0 (Fake)
        results = []
        for word, weight in contributions:
            results.append({
                "word": word,
                "weight": weight,
                "impact": "real" if weight > 0 else "fake"  # Fixed the logic here
            })

        response_data = {
            "review": review,
            "top_words": results,
            "model_type": MODEL_TYPE
        }

        print(f"✅ Explanation generated successfully with {len(results)} words")
        return jsonify(response_data)

    except Exception as e:
        print(f"❌ Error in explain: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Explanation failed: {str(e)}"}), 500


@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    status = {
        "status": "healthy",
        "model_type": MODEL_TYPE,
        "model_loaded": False,
        "explainer_loaded": False
    }
    
    if MODEL_TYPE == "logreg":
        status["model_loaded"] = pipeline is not None
        status["explainer_loaded"] = explainer is not None
    elif MODEL_TYPE == "bert":
        status["model_loaded"] = 'model' in globals() and 'tokenizer' in globals()
        
    return jsonify(status)


if __name__ == "__main__":
    print(f"🚀 Starting server with {MODEL_TYPE.upper()} model...")
    app.run(debug=True, port=5000, host="0.0.0.0")