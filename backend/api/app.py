from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import joblib
import os
import numpy as np
import io
import base64
import matplotlib
matplotlib.use('Agg')  # Non-GUI backend for server
import matplotlib.pyplot as plt
import seaborn as sns

# Hugging Face imports for BERT
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import shap

app = Flask(__name__)
CORS(app)

# ============================
# CONFIG: choose model type
# ============================
MODEL_TYPE = "logreg"   # options: "logreg"

# ============================
# Load models
# ============================
pipeline = None
explainer = None
logreg_pipeline = None
svm_pipeline = None
tuned_pipeline = None
voting_ensemble = None

if MODEL_TYPE == "logreg":
    MODEL_DIR = os.path.join(os.path.dirname(__file__), "../model/results")
    
    # Load main pipeline
    MAIN_MODEL_PATH = os.path.join(MODEL_DIR, "review_pipeline.pkl")
    SHAP_PATH = os.path.join(MODEL_DIR, "shap_explainer.pkl")
    
    # Load additional models
    LOGREG_PATH = os.path.join(MODEL_DIR, "logreg_pipeline.pkl")
    SVM_PATH = os.path.join(MODEL_DIR, "svm_pipeline.pkl")
    TUNED_PATH = os.path.join(MODEL_DIR, "tuned_pipeline.pkl")
    
    try:
        # Load main pipeline (could be ensemble or tuned model)
        pipeline = joblib.load(MAIN_MODEL_PATH)
        print("✅ Main model loaded successfully!")
        
        # Load SHAP explainer
        try:
            explainer = joblib.load(SHAP_PATH)
            print("✅ SHAP explainer loaded successfully!")
        except:
            print("⚠️ SHAP explainer not found, explanations will be limited")
        
        # Load individual models for comparison
        try:
            logreg_pipeline = joblib.load(LOGREG_PATH)
            print("✅ Logistic Regression model loaded!")
        except:
            print("⚠️ LogReg model not found")
            
        try:
            svm_pipeline = joblib.load(SVM_PATH)
            print("✅ SVM model loaded!")
        except:
            print("⚠️ SVM model not found")
            
        try:
            tuned_pipeline = joblib.load(TUNED_PATH)
            print("✅ Tuned model loaded!")
        except:
            print("⚠️ Tuned model not found")
            
    except Exception as e:
        print(f"❌ Error loading models: {e}")

elif MODEL_TYPE == "bert":
    MODEL_PATH = os.path.join(os.path.dirname(__file__), "../model/bert_model")
    try:
        tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
        model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
        print("✅ BERT model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading BERT model: {e}")

# ============================
# Helper Functions
# ============================
def fig_to_base64(fig):
    """Convert matplotlib figure to base64 string"""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_base64

# ============================
# Routes
# ============================
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Enhanced Fake Review Detection API",
        "version": "2.0",
        "model_type": MODEL_TYPE,
        "features": [
            "Multi-model ensemble prediction",
            "SHAP explanations",
            "Confidence visualization",
            "Model comparison",
            "Real-time analysis"
        ]
    })


@app.route("/predict", methods=["POST"])
def predict():
    """Main prediction endpoint with enhanced response"""
    try:
        data = request.get_json(silent=True) or {}
        review = data.get("review", "").strip()

        if not review:
            return jsonify({"error": "No review text provided"}), 400

        if MODEL_TYPE == "logreg":
            if pipeline is None:
                return jsonify({"error": "Model not loaded"}), 500
                
            # Main prediction
            pred = pipeline.predict([review])[0]
            proba = pipeline.predict_proba([review])[0].tolist()
            
            # Get predictions from all available models
            model_predictions = {
                "main_model": {
                    "prediction": int(pred),
                    "confidence": float(max(proba)),
                    "probabilities": {"fake": float(proba[0]), "real": float(proba[1])}
                }
            }
            
            # Add individual model predictions if available
            if logreg_pipeline:
                lr_pred = logreg_pipeline.predict([review])[0]
                lr_proba = logreg_pipeline.predict_proba([review])[0]
                model_predictions["logistic_regression"] = {
                    "prediction": int(lr_pred),
                    "confidence": float(max(lr_proba)),
                    "probabilities": {"fake": float(lr_proba[0]), "real": float(lr_proba[1])}
                }
            
            if svm_pipeline:
                svm_pred = svm_pipeline.predict([review])[0]
                # SVM decision function for confidence
                svm_decision = svm_pipeline.decision_function([review])[0]
                svm_confidence = 1 / (1 + np.exp(-abs(svm_decision)))  # Convert to probability-like score
                model_predictions["svm"] = {
                    "prediction": int(svm_pred),
                    "confidence": float(svm_confidence),
                    "decision_score": float(svm_decision)
                }
            
            if tuned_pipeline and tuned_pipeline != pipeline:
                tuned_pred = tuned_pipeline.predict([review])[0]
                tuned_proba = tuned_pipeline.predict_proba([review])[0]
                model_predictions["tuned_model"] = {
                    "prediction": int(tuned_pred),
                    "confidence": float(max(tuned_proba)),
                    "probabilities": {"fake": float(tuned_proba[0]), "real": float(tuned_proba[1])}
                }
            
            # Calculate consensus
            predictions_list = [m["prediction"] for m in model_predictions.values() if "prediction" in m]
            consensus = sum(predictions_list) / len(predictions_list) if predictions_list else pred
            agreement = (sum(1 for p in predictions_list if p == pred) / len(predictions_list) * 100) if predictions_list else 100
            
            # Confidence level assessment
            confidence_level = "High" if max(proba) > 0.85 else "Medium" if max(proba) > 0.70 else "Low"
            
            response = {
                "prediction": int(pred),
                "result": "Real Review ✅" if pred == 1 else "Fake Review ❌",
                "confidence": float(max(proba)),
                "confidence_level": confidence_level,
                "probability": {
                    "fake": float(proba[0]), 
                    "real": float(proba[1])
                },
                "review": review,
                "model_predictions": model_predictions,
                "consensus_score": float(consensus),
                "model_agreement": f"{agreement:.1f}%",
                "review_stats": {
                    "length": len(review),
                    "word_count": len(review.split()),
                    "avg_word_length": np.mean([len(word) for word in review.split()]) if review.split() else 0,
                    "exclamation_count": review.count('!'),
                    "question_count": review.count('?'),
                    "uppercase_ratio": sum(1 for c in review if c.isupper()) / len(review) if review else 0
                }
            }

        elif MODEL_TYPE == "bert":
            inputs = tokenizer(review, return_tensors="pt", truncation=True, padding=True, max_length=128)
            with torch.no_grad():
                outputs = model(**inputs)
                probs = torch.softmax(outputs.logits, dim=1).tolist()[0]
            pred = int(torch.argmax(torch.tensor(probs)))
            
            response = {
                "prediction": int(pred),
                "result": "Real Review ✅" if pred == 1 else "Fake Review ❌",
                "confidence": float(max(probs)),
                "probability": {"fake": float(probs[0]), "real": float(probs[1])},
                "review": review
            }

        return jsonify(response)

    except Exception as e:
        print(f"Error in predict: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


@app.route("/explain", methods=["POST"])
def explain():
    """Enhanced explanation with SHAP values and visualization"""
    try:
        if MODEL_TYPE != "logreg":
            return jsonify({"error": "Explanation available only for Logistic Regression model"}), 400

        if pipeline is None:
            return jsonify({"error": "Model not loaded"}), 500

        data = request.get_json(silent=True) or {}
        review = data.get("review", "").strip()
        
        if not review:
            return jsonify({"error": "No review text provided"}), 400

        print(f"🔍 Explaining review: {review[:100]}...")

        # Get the correct pipeline for explanation (prefer one with named_steps)
        explain_pipeline = logreg_pipeline if logreg_pipeline else pipeline
        
        if not hasattr(explain_pipeline, 'named_steps'):
            return jsonify({"error": "Pipeline structure not compatible"}), 500

        # Transform the review using TF-IDF
        tfidf_matrix = explain_pipeline.named_steps["tfidf"].transform([review])
        
        # Get prediction and probability
        pred = explain_pipeline.predict([review])[0]
        proba = explain_pipeline.predict_proba([review])[0]
        
        # Get SHAP values if explainer available
        word_importance = []
        
        if explainer:
            try:
                shap_values = explainer(tfidf_matrix)
                
                if hasattr(shap_values, 'values'):
                    values_array = shap_values.values[0]
                else:
                    values_array = shap_values[0]

                feature_names = explain_pipeline.named_steps["tfidf"].get_feature_names_out()
                
                # Create word-importance pairs
                for word, importance in zip(feature_names, values_array):
                    if abs(importance) > 1e-6:
                        word_importance.append({
                            "word": word,
                            "weight": float(importance),
                            "impact": "real" if importance > 0 else "fake",
                            "abs_weight": abs(float(importance))
                        })
                
                # Sort by absolute importance
                word_importance = sorted(word_importance, key=lambda x: x['abs_weight'], reverse=True)[:20]
                
            except Exception as e:
                print(f"⚠️ SHAP explanation failed: {e}")
        
        # If SHAP not available, use coefficients
        if not word_importance:
            try:
                feature_names = explain_pipeline.named_steps["tfidf"].get_feature_names_out()
                coefficients = explain_pipeline.named_steps["clf"].coef_[0]
                
                # Get non-zero features in this review
                feature_indices = tfidf_matrix.nonzero()[1]
                
                for idx in feature_indices:
                    word = feature_names[idx]
                    coef = coefficients[idx]
                    tfidf_value = tfidf_matrix[0, idx]
                    importance = coef * tfidf_value
                    
                    word_importance.append({
                        "word": word,
                        "weight": float(importance),
                        "impact": "real" if importance > 0 else "fake",
                        "abs_weight": abs(float(importance))
                    })
                
                word_importance = sorted(word_importance, key=lambda x: x['abs_weight'], reverse=True)[:20]
                
            except Exception as e:
                print(f"⚠️ Coefficient explanation failed: {e}")

        # Separate positive and negative contributions
        positive_words = [w for w in word_importance if w['weight'] > 0][:10]
        negative_words = [w for w in word_importance if w['weight'] < 0][:10]

        response_data = {
            "review": review,
            "prediction": int(pred),
            "result": "Real Review ✅" if pred == 1 else "Fake Review ❌",
            "confidence": float(max(proba)),
            "probability": {"fake": float(proba[0]), "real": float(proba[1])},
            "top_words": word_importance[:15],
            "positive_indicators": positive_words,
            "negative_indicators": negative_words,
            "model_type": MODEL_TYPE,
            "explanation_method": "SHAP" if explainer else "Coefficients"
        }

        print(f"✅ Explanation generated with {len(word_importance)} important words")
        return jsonify(response_data)

    except Exception as e:
        print(f"❌ Error in explain: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Explanation failed: {str(e)}"}), 500


@app.route("/visualize", methods=["POST"])
def visualize():
    """Generate visualization for a prediction"""
    try:
        data = request.get_json(silent=True) or {}
        review = data.get("review", "").strip()
        
        if not review or pipeline is None:
            return jsonify({"error": "Invalid request"}), 400
        
        # Get prediction
        pred = pipeline.predict([review])[0]
        proba = pipeline.predict_proba([review])[0]
        
        # Create confidence visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot 1: Probability bar chart
        labels = ['Fake', 'Real']
        colors = ['#e74c3c', '#2ecc71']
        bars = ax1.barh(labels, proba, color=colors, alpha=0.7, edgecolor='black')
        ax1.set_xlabel('Probability')
        ax1.set_title('Prediction Confidence')
        ax1.set_xlim([0, 1])
        
        # Add value labels
        for bar, prob in zip(bars, proba):
            width = bar.get_width()
            ax1.text(width, bar.get_y() + bar.get_height()/2, 
                    f'{prob:.1%}', ha='left', va='center', fontweight='bold')
        
        # Plot 2: Model comparison (if multiple models available)
        model_names = []
        model_preds = []
        
        if logreg_pipeline:
            model_names.append('LogReg')
            model_preds.append(logreg_pipeline.predict_proba([review])[0][1])
        
        if svm_pipeline:
            model_names.append('SVM')
            svm_pred = svm_pipeline.predict([review])[0]
            model_preds.append(float(svm_pred))
        
        if tuned_pipeline:
            model_names.append('Tuned')
            model_preds.append(tuned_pipeline.predict_proba([review])[0][1])
        
        if model_names:
            colors_models = ['#3498db', '#e74c3c', '#2ecc71']
            ax2.bar(model_names, model_preds, color=colors_models[:len(model_names)], alpha=0.7, edgecolor='black')
            ax2.set_ylabel('Real Review Probability')
            ax2.set_title('Model Comparison')
            ax2.set_ylim([0, 1])
            ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
            
            # Add value labels
            for i, (name, pred_val) in enumerate(zip(model_names, model_preds)):
                ax2.text(i, pred_val, f'{pred_val:.2f}', ha='center', va='bottom', fontweight='bold')
        else:
            ax2.text(0.5, 0.5, 'Single Model', ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Model Comparison (N/A)')
        
        plt.tight_layout()
        
        # Convert to base64
        img_base64 = fig_to_base64(fig)
        
        return jsonify({
            "image": f"data:image/png;base64,{img_base64}",
            "prediction": int(pred),
            "confidence": float(max(proba))
        })
        
    except Exception as e:
        print(f"❌ Error in visualize: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Visualization failed: {str(e)}"}), 500


@app.route("/batch-predict", methods=["POST"])
def batch_predict():
    """Predict multiple reviews at once"""
    try:
        data = request.get_json(silent=True) or {}
        reviews = data.get("reviews", [])
        
        if not reviews or not isinstance(reviews, list):
            return jsonify({"error": "Please provide a list of reviews"}), 400
        
        if len(reviews) > 100:
            return jsonify({"error": "Maximum 100 reviews per batch"}), 400
        
        results = []
        
        for review in reviews:
            review = review.strip()
            if not review:
                continue
                
            pred = pipeline.predict([review])[0]
            proba = pipeline.predict_proba([review])[0]
            
            results.append({
                "review": review,
                "prediction": int(pred),
                "result": "Real" if pred == 1 else "Fake",
                "confidence": float(max(proba)),
                "probability": {"fake": float(proba[0]), "real": float(proba[1])}
            })
        
        # Summary statistics
        fake_count = sum(1 for r in results if r['prediction'] == 0)
        real_count = sum(1 for r in results if r['prediction'] == 1)
        avg_confidence = np.mean([r['confidence'] for r in results])
        
        return jsonify({
            "results": results,
            "summary": {
                "total": len(results),
                "fake": fake_count,
                "real": real_count,
                "fake_percentage": f"{fake_count/len(results)*100:.1f}%",
                "average_confidence": f"{avg_confidence:.2%}"
            }
        })
        
    except Exception as e:
        print(f"❌ Error in batch_predict: {e}")
        return jsonify({"error": f"Batch prediction failed: {str(e)}"}), 500


@app.route("/model-info", methods=["GET"])
def model_info():
    """Get information about loaded models"""
    try:
        info = {
            "model_type": MODEL_TYPE,
            "models_loaded": {
                "main_pipeline": pipeline is not None,
                "logistic_regression": logreg_pipeline is not None,
                "svm": svm_pipeline is not None,
                "tuned_model": tuned_pipeline is not None,
                "shap_explainer": explainer is not None
            },
            "capabilities": [
                "Single prediction",
                "Batch prediction",
                "SHAP explanations" if explainer else None,
                "Multi-model comparison" if logreg_pipeline or svm_pipeline else None,
                "Confidence visualization"
            ]
        }
        
        # Remove None values
        info["capabilities"] = [c for c in info["capabilities"] if c]
        
        # Get model statistics if available
        if hasattr(pipeline, 'named_steps') and hasattr(pipeline.named_steps.get('tfidf'), 'vocabulary_'):
            info["vocabulary_size"] = len(pipeline.named_steps['tfidf'].vocabulary_)
        
        return jsonify(info)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    status = {
        "status": "healthy" if pipeline else "degraded",
        "model_type": MODEL_TYPE,
        "models": {
            "main_pipeline": pipeline is not None,
            "logistic_regression": logreg_pipeline is not None,
            "svm": svm_pipeline is not None,
            "tuned_model": tuned_pipeline is not None,
            "explainer": explainer is not None
        },
        "version": "2.0",
        "features": ["predict", "explain", "visualize", "batch-predict"]
    }
    
    return jsonify(status)


if __name__ == "__main__":
    print(f"🚀 Starting Enhanced API with {MODEL_TYPE.upper()} model...")
    print(f"📊 Models loaded: Pipeline={pipeline is not None}, LogReg={logreg_pipeline is not None}, SVM={svm_pipeline is not None}")
    print(f"🔍 SHAP explainer: {explainer is not None}")
    app.run(debug=True, port=8000, host="0.0.0.0")