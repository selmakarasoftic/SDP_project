# SDP_project

# Customer Review Analysis System

## Overview
This project presents a system for analyzing customer reviews using Natural Language Processing (NLP) and Machine Learning techniques. The goal of the system is to automatically extract meaningful insights from customer feedback and support decision-making in customer support management.

## Features
The system performs the following tasks:
- Sentiment classification (positive, negative, neutral)
- Complaint priority detection (low, medium, high)
- Decision on handling type (automated system or human agent)
- Complaint category classification (e.g. delivery, packaging, product quality)

## Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- NLTK
- VADER Sentiment Analyzer

## Methodology
The system follows a structured NLP pipeline:
1. Dataset loading and preprocessing
2. Text cleaning (removing noise, stopwords, handling negations)
3. Feature extraction using TF-IDF
4. Sentiment classification using Logistic Regression
5. Rule-based layers for:
   - priority assignment
   - human intervention decision
   - complaint categorization

## Example Output
For a given customer review, the system provides:
- Predicted sentiment
- Complaint priority level
- Handling type (agent or automated)
- Complaint category

## Dataset
The system uses the Amazon Fine Food Reviews dataset:
https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews

## libraries that need to be installed in terminal
pip install pandas numpy scikit-learn nltk matplotlib
pip install joblib
pip install vaderSentiment
