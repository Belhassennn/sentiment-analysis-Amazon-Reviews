# 📊 Sentiment Analysis on Amazon Fine Food Reviews

This project performs sentiment analysis on the **Amazon Fine Food Reviews** dataset using a combination of traditional NLP techniques and transformer models such as **VADER** and **RoBERTa**. The goal is to classify the sentiment of customer reviews as positive, neutral, or negative.

## 📁 Table of Contents
- [📚 Project Overview](#-project-overview)
- [📊 Dataset](#-dataset)
- [🔧 Preprocessing](#-preprocessing)
- [🤖 Modeling](#-modeling)
  - [Naive Bayes](#naive-bayes)
  - [VADER Sentiment Analysis](#vader-sentiment-analysis)
  - [RoBERTa Transformer](#roberta-transformer)
- [📈 Results](#-results)
- [🔮 Future Work](#-future-work)
- [🚀 How to Run](#-how-to-run)

## 📚 Project Overview

Sentiment analysis helps businesses understand customer feedback by determining the sentiment (positive, neutral, or negative) of product reviews. In this project, we focus on sentiment analysis using both rule-based and machine learning models.

## 📊 Dataset

The dataset used in this project is the **Amazon Fine Food Reviews** dataset, which contains customer reviews for various food products sold on Amazon. The dataset includes:
- **Text**: The actual review text.
- **Score**: Sentiment score of the review, ranging from 1 to 5.
- **Id**: Unique identifier for each review.

## 🔧 Preprocessing

The preprocessing steps include:
- **Tokenization**: Splitting text into individual words.
- **Removing Stopwords**: Elimination of common words that do not contribute to sentiment.
- **TF-IDF Vectorization**: Converting the text into numerical format.
- **Lemmatization**: Reducing words to their root form.

## 🤖 Modeling

This project includes multiple models for sentiment analysis:

### Naive Bayes
We used **Multinomial Naive Bayes** to classify the sentiment of reviews. The model was trained on TF-IDF vectors of the reviews.

### VADER Sentiment Analysis
**VADER (Valence Aware Dictionary and Sentiment Reasoner)** is a rule-based model for general sentiment analysis. It provides a positive, neutral, and negative sentiment score for each review.

### RoBERTa Transformer
We used a **RoBERTa (Robustly Optimized BERT Pretraining Approach)** transformer model to analyze the sentiment of reviews. This is a deep learning approach leveraging transfer learning to capture complex language patterns.


## 🔮 Future Work

There are several areas for potential improvement:
- **Hyperparameter tuning** for the Naive Bayes and RoBERTa models.
- **Feature engineering** to add more relevant features to the model.
- **Model ensembling**: Combine different models to improve overall performance.
- **Exploration of other transformer models** such as BERT or GPT
