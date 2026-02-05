# FAKE REVIEW DETECTION SYSTEM

## Overview
This project is a full-stack web application that detects fake Amazon reviews using machine learning models. It features a React frontend and a Flask backend, with models trained on real Amazon review data.

## Features
- Analyze Amazon reviews for authenticity
- Probability and explanation of review classification
- Modern, responsive UI with dark mode
- RESTful API backend
- Machine learning models (Logistic Regression, SVM, SHAP explanations)

## Project Structure
```
FAKE-REVIEW-DETECTION-SYSTEM/
├── backend/
│   ├── api/
│   │   ├── app.py
│   │   ├── app1.py
│   │   ├── requirements.txt
│   │   └── __pycache__/
│   └── model/
│       ├── train_model.py
│       ├── train_model1.py
│       └── results/
├── dataset/
│   └── amazon_reviews_2018.csv
├── frontend/
│   ├── index.html
│   ├── package.json
│   ├── postcss.config.js
│   ├── tailwind.config.js
│   ├── vite.config.js
│   └── src/
│       ├── App.jsx
│       ├── index.css
│       ├── main.jsx
│       └── components/
│           ├── DarkModeToggle.jsx
│           ├── Header.jsx
│           ├── ProbabilityBar.jsx
│           ├── ResultCard.jsx
│           ├── ReviewForm.jsx
│           └── Toast.jsx
│       └── utils/
│           └── api.js
└── README.md
```

## Getting Started

### Prerequisites
- Python 3.14+
- Node.js 18+
- npm

### Backend Setup
```bash
cd backend/api
python -m venv ../../.venv
source ../../.venv/bin/activate
pip install -r requirements.txt
python app.py
```

### Frontend Setup
```bash
cd frontend
npm install
npm run dev
```

### Usage
- Open your browser and go to the frontend server URL (e.g., http://localhost:3001)
- Enter an Amazon review and analyze it

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License
This project is licensed under the MIT License.

## Author
- [harshbadhann2](https://github.com/harshbadhann2)