FAKE-REVIEW-DETECTION-SYSTEM
A full-stack machine learning application that detects fake Amazon reviews using advanced NLP techniques and provides explainable AI insights.
Overview
This project combines machine learning, natural language processing, and explainable AI to identify fraudulent Amazon reviews. The system uses multiple ML models including Logistic Regression, SVM, and BERT to analyze review authenticity, while SHAP (SHapley Additive exPlanations) provides transparency into model decisions. The application features a modern React frontend with interactive visualizations and a robust Flask backend API.
Features

Multi-Model Analysis: Leverages Logistic Regression, SVM, and BERT models for comprehensive review analysis
Explainable AI: SHAP-based explanations help users understand why a review was flagged as fake or genuine
Interactive Dashboard: Modern React frontend built with Vite and TailwindCSS
Probability Visualization: Clear visual representation of prediction confidence scores
RESTful API: Well-documented endpoints for easy integration
Real-time Predictions: Fast analysis of Amazon review text

Project Structure
FAKE-REVIEW-DETECTION-SYSTEM/
├── frontend/                 # React frontend application
│   ├── src/                 # Source files
│   ├── public/              # Static assets
│   ├── package.json         # Frontend dependencies
│   └── vite.config.js       # Vite configuration
├── backend/
│   └── api/                 # Flask API server
│       ├── app.py           # Main application entry point
│       ├── requirements.txt # Python dependencies
│       └── ...              # API modules
├── model/                   # Model training and results
│   ├── train_model.py       # Primary training script
│   ├── train_model1.py      # Alternative training script
│   └── ...                  # Saved models and artifacts
├── dataset/                 # Amazon reviews dataset
│   └── ...                  # Training and test data
├── LICENSE                  # MIT License
└── README.md               # This file
Setup
Prerequisites

Backend: Python 3.10 or higher
Frontend: Node.js 18 or higher
Package Manager: npm or yarn

Backend Setup

Navigate to the backend directory:

bash   cd backend/api

Create and activate a virtual environment (recommended):

bash   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate

Install dependencies:

bash   pip install -r requirements.txt

Run the Flask server:

bash   python app.py
The API will start on http://localhost:5000 by default.
Frontend Setup

Navigate to the frontend directory:

bash   cd frontend

Install dependencies:

bash   npm install

Start the development server:

bash   npm run dev
The application will open at http://localhost:5173 (or another port if 5173 is busy).
Usage

Start both servers: Ensure both the backend Flask API and frontend development server are running.
Access the application: Open your browser and navigate to the frontend URL (typically http://localhost:5173).
Analyze a review:

Paste or type an Amazon review into the input field
Click the "Analyze" button
View the prediction (Fake/Real) along with probability scores
Explore SHAP explanations to understand the model's decision


Interpret results:

Prediction: Shows whether the review is likely fake or genuine
Probability: Displays confidence level (0-100%)
Explanation: Highlights words and features that influenced the decision



API Endpoints
POST /predict
Get a fake/real prediction for a given review.
Request Body:
json{
  "review_text": "This product is amazing! Highly recommend it!"
}
Response:
json{
  "prediction": "Real",
  "probability": 0.87,
  "model": "BERT"
}
POST /explain
Get SHAP-based explanation for a review prediction.
Request Body:
json{
  "review_text": "This product is amazing! Highly recommend it!"
}
Response:
json{
  "explanation": {
    "feature_importance": [...],
    "shap_values": [...],
    "base_value": 0.5
  }
}
POST /visualize
Get probability visualization data.
Request Body:
json{
  "review_text": "This product is amazing! Highly recommend it!"
}
Response:
json{
  "visualization": {
    "fake_probability": 0.13,
    "real_probability": 0.87,
    "chart_data": [...]
  }
}
Model Training
The model/ directory contains scripts for training and evaluating machine learning models.
Training a New Model

Navigate to the model directory:

bash   cd model

Run the training script:

bash   python train_model.py
Or use the alternative training script:
bash   python train_model1.py

Training outputs:

Trained model files (saved in model/ directory)
Performance metrics and evaluation reports
Model comparison results



Available Models

Logistic Regression: Fast, interpretable baseline model
SVM (Support Vector Machine): Robust classification with kernel tricks
BERT: State-of-the-art transformer-based language model

Contributing
Contributions are welcome! Here's how you can help:

Fork the repository
Create a feature branch:

bash   git checkout -b feature/your-feature-name

Make your changes and commit:

bash   git commit -m "Add your meaningful commit message"

Push to your branch:

bash   git push origin feature/your-feature-name

Open a Pull Request: Provide a clear description of your changes

Guidelines

Open an issue first for major changes to discuss your ideas
Follow existing code style and conventions
Add tests for new features when applicable
Update documentation as needed
Ensure all tests pass before submitting PR

License
This project is licensed under the MIT License. See the LICENSE file for details.

Acknowledgments

Dataset sourced from Amazon review datasets
Built with Flask, React, TailwindCSS, and Vite
SHAP library for explainable AI
BERT model from Hugging Face Transformers

Support
If you encounter any issues or have questions:

Open an issue on GitHub
Check existing documentation
Review closed issues for similar problems
