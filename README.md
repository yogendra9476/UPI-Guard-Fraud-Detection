# UPI Guard: Real-Time Intelligent UPI Fraud Detection Using 1D-CNN

An end-to-end Machine Learning system engineered for real-time UPI transaction screening and pre-debit fraudulent transaction blocking using 1D Convolutional Neural Networks.

## Team Details (Goa Institute of Management - PGDM Big Data Analytics)
* **Yogendra Pandey** (ID: B2026109)
* **Sahil Mangla** (ID: B2026097)
* **Kuldeep Khengar** (ID: B2026082)

## Key Highlights
* **Pre-Debit Interception:** Evaluates transaction risk dynamically before fund settlement.
* **Architecture:** 1D-CNN trained on spatio-temporal, demographic, and merchant domain features.
* **Benchmark Accuracy:** Achieves 99.6% classification accuracy, surpassing baseline models (SVM, Random Forest, Logistic Regression).
* **Multi-Portal Application:** Flask-based dashboard supporting Admin, Merchant, and End-User roles with OTP-verified access.

## Project Structure
```text
UPI-Guard-Fraud-Detection/
├── dataset/              # Raw and processed UPI transaction datasets
├── notebooks/            # Exploratory Data Analysis & baseline benchmarking
├── src/
│   ├── feature_pipeline.py  # Encoding, normalization, and reshaping
│   └── train_model.py       # 1D-CNN training and serialization (.h5)
├── app/
│   ├── templates/        # HTML views for Admin, Merchant, and User
│   ├── static/           # CSS, JS, and dynamically generated QR codes
│   └── app.py            # Flask application gateway
├── requirements.txt      # Environment dependencies
└── README.md
