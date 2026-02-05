# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.linear_model import LogisticRegression
# from sklearn.pipeline import Pipeline
# from sklearn.metrics import classification_report, accuracy_score
# import joblib
# import shap
# import os

# # Hugging Face imports for BERT
# from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
# from torch.utils.data import Dataset
# import torch

# # ============================
# # CONFIG: choose model type
# # ============================
# MODEL_TYPE = "logreg"   # options: "logreg", "bert"

# # ============================
# # Load dataset
# # ============================
# df = pd.read_csv("../../dataset/amazon_reviews_2018.csv")
# df = df.rename(columns={"text_": "review", "label": "label"})
# df["label"] = df["label"].map({"OR": 1, "CG": 0})

# X = df["review"].astype(str)
# y = df["label"]

# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42, stratify=y
# )

# # ============================
# # Logistic Regression pipeline
# # ============================
# if MODEL_TYPE == "logreg":
#     pipeline = Pipeline([
#         ("tfidf", TfidfVectorizer(
#             stop_words="english",
#             max_features=10000,
#             ngram_range=(1, 2)
#         )),
#         ("clf", LogisticRegression(max_iter=300, class_weight="balanced"))
#     ])

#     pipeline.fit(X_train, y_train)
#     y_pred = pipeline.predict(X_test)

#     print("✅ Accuracy:", accuracy_score(y_test, y_pred))
#     print("\nClassification Report:\n",
#           classification_report(y_test, y_pred, target_names=["Fake (CG)", "Real (OR)"]))

#     joblib.dump(pipeline, "review_pipeline.pkl")
#     print("✅ Logistic Regression pipeline saved successfully")

#     # Save a SHAP explainer (works with linear models)
#     explainer = shap.Explainer(pipeline.named_steps["clf"],
#                                pipeline.named_steps["tfidf"].transform(X_train))
#     joblib.dump(explainer, "shap_explainer.pkl")
#     print("✅ SHAP explainer saved successfully")


# # ============================
# # BERT fine-tuning pipeline
# # ============================
# if MODEL_TYPE == "bert":

#     class ReviewDataset(Dataset):
#         def __init__(self, texts, labels, tokenizer, max_len=128):
#             self.encodings = tokenizer(list(texts), truncation=True, padding=True, max_length=max_len)
#             self.labels = list(labels)

#         def __len__(self):
#             return len(self.labels)

#         def __getitem__(self, idx):
#             item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
#             item["labels"] = torch.tensor(self.labels[idx])
#             return item

#     tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
#     model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

#     train_dataset = ReviewDataset(X_train, y_train, tokenizer)
#     test_dataset = ReviewDataset(X_test, y_test, tokenizer)

#     training_args = TrainingArguments(
#         output_dir="./results",
#         evaluation_strategy="epoch",
#         save_strategy="epoch",
#         num_train_epochs=2,
#         per_device_train_batch_size=16,
#         per_device_eval_batch_size=16,
#         learning_rate=2e-5,
#         weight_decay=0.01,
#         logging_dir="./logs",
#     )

#     trainer = Trainer(
#         model=model,
#         args=training_args,
#         train_dataset=train_dataset,
#         eval_dataset=test_dataset,
#     )

#     trainer.train()
#     trainer.save_model("./bert_model")
#     tokenizer.save_pretrained("./bert_model")

#     print("✅ BERT model fine-tuned and saved successfully")











import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
import shap
import torch

from sklearn.model_selection import train_test_split, learning_curve
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve
)

from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments
)
from torch.utils.data import Dataset

# ===========================================
# CONFIG
# ===========================================
MODEL_TYPE = "bert"   # options: "logreg", "bert"
DATA_PATH = "../../dataset/amazon_reviews_2018.csv"
RESULTS_DIR = "./results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ===========================================
# LOAD DATA
# ===========================================
df = pd.read_csv(DATA_PATH)
df = df.rename(columns={"text_": "review", "label": "label"})
df["label"] = df["label"].map({"OR": 1, "CG": 0})
df = df.dropna(subset=["review"])

X = df["review"].astype(str)
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ===========================================
# LOGISTIC REGRESSION PIPELINE
# ===========================================
if MODEL_TYPE == "logreg":
    print("🚀 Training Logistic Regression model...")

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            stop_words="english",
            max_features=10000,
            ngram_range=(1, 2)
        )),
        ("clf", LogisticRegression(max_iter=300, class_weight="balanced"))
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_pred_prob = pipeline.predict_proba(X_test)[:, 1]

    # ==== Metrics ====
    acc = accuracy_score(y_test, y_pred)
    print("✅ Accuracy:", acc)
    print("\nClassification Report:\n",
          classification_report(y_test, y_pred, target_names=["Fake (CG)", "Real (OR)"]))

    # ==== Confusion Matrix ====
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Fake", "Real"], yticklabels=["Fake", "Real"])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

    # ==== ROC Curve ====
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.show()

    # ==== Precision-Recall Curve ====
    precision, recall, _ = precision_recall_curve(y_test, y_pred_prob)
    plt.plot(recall, precision, label="Precision-Recall curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.show()

    # ==== Learning Curve ====
    train_sizes, train_scores, test_scores = learning_curve(
        pipeline, X_train, y_train, cv=5, scoring="accuracy",
        train_sizes=np.linspace(0.1, 1.0, 5)
    )
    train_mean = np.mean(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)

    plt.plot(train_sizes, train_mean, 'o-', label="Training score")
    plt.plot(train_sizes, test_mean, 'o-', label="Validation score")
    plt.xlabel("Training Set Size")
    plt.ylabel("Accuracy")
    plt.title("Learning Curve")
    plt.legend()
    plt.show()

    # ==== Save Model & SHAP ====
    joblib.dump(pipeline, os.path.join(RESULTS_DIR, "review_pipeline.pkl"))
    print("✅ Model saved successfully.")

    explainer = shap.Explainer(pipeline.named_steps["clf"],
                               pipeline.named_steps["tfidf"].transform(X_train))
    joblib.dump(explainer, os.path.join(RESULTS_DIR, "shap_explainer.pkl"))
    print("✅ SHAP explainer saved successfully.")


# ===========================================
# BERT FINE-TUNING PIPELINE
# ===========================================
if MODEL_TYPE == "bert":
    print("🚀 Fine-tuning BERT model...")

    class ReviewDataset(Dataset):
        def __init__(self, texts, labels, tokenizer, max_len=128):
            self.encodings = tokenizer(list(texts), truncation=True, padding=True, max_length=max_len)
            self.labels = list(labels)

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item["labels"] = torch.tensor(self.labels[idx])
            return item

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

    train_dataset = ReviewDataset(X_train, y_train, tokenizer)
    test_dataset = ReviewDataset(X_test, y_test, tokenizer)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=1)
        acc = accuracy_score(labels, preds)
        return {"accuracy": acc}

    training_args = TrainingArguments(
        output_dir=RESULTS_DIR,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=2,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        learning_rate=2e-5,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        load_best_model_at_end=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(os.path.join(RESULTS_DIR, "bert_model"))
    tokenizer.save_pretrained(os.path.join(RESULTS_DIR, "bert_model"))

    print("✅ BERT model fine-tuned and saved successfully.")

    # ==== Plot Training Curves ====
    logs = trainer.state.log_history
    train_loss = [x["loss"] for x in logs if "loss" in x]
    eval_loss = [x["eval_loss"] for x in logs if "eval_loss" in x]
    eval_acc = [x["eval_accuracy"] for x in logs if "eval_accuracy" in x]

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_loss, label="Train Loss")
    plt.plot(eval_loss, label="Eval Loss")
    plt.title("Loss over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(eval_acc, label="Eval Accuracy", color="green")
    plt.title("Evaluation Accuracy over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.show()