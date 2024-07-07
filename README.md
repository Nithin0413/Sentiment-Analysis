# Sentiment Analysis Using Naive Bayes and SVM #
## Overview ##
This project focuses on sentiment analysis using machine learning techniques, specifically Multinomial Naive Bayes and Support Vector Machine (SVM) classifiers. The goal is to classify the sentiment (positive, negative, neutral) of tweets from a dataset using natural language processing (NLP) techniques.

## Dataset ##
The dataset (tweets.csv) contains tweets with associated sentiment labels (airline_sentiment). It includes preprocessing steps such as tokenization, stopwords removal, lemmatization, and feature extraction.

## Approach ##
### Data Preprocessing: ###

* Read and preprocess the tweet data (tweets.csv).
* Tokenize tweets using word_tokenize from NLTK.
* Remove stopwords and punctuation marks.
* Lemmatize words to reduce them to their base form.
* Split data into training and evaluation sets using train_test_split.
  
### Feature Extraction: ###
* Construct a vocabulary of words excluding stopwords and infrequent words.
* Extract features using:
* Direct word counts with customized word features.
* Utilize CountVectorizer from scikit-learn for feature extraction.
  
### Model Training: ###
* Train a Multinomial Naive Bayes classifier (MultinomialNB) using:
* Handcrafted word features.
* Features extracted via CountVectorizer.
* Train an SVM classifier (SVC) with a linear kernel for comparison.

### Evaluation: ###
* Evaluate models based on accuracy scores using metrics.accuracy_score.
* Visualize word frequency distribution using matplotlib.
  
### Output: ###
* Predict sentiment labels (positive, negative, neutral) for evaluation data.
* Display accuracy metrics and visualizations of word frequencies.
