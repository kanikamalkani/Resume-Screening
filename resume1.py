# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 23:38:01 2023

@author: kanik
"""

import pandas as pd
import numpy as np
import spacy
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# Step 1: Data preprocessing
nlp = spacy.load('en_core_web_sm')
# Define functions to preprocess resume text data, e.g. using NLP techniques

# Step 2: Define job requirements
job_requirements = ['education', 'work experience', 'skills']

# Step 3: Feature extraction
count_vect = CountVectorizer()
tfidf_transformer = TfidfTransformer()

# Define functions to extract relevant features from resume text data

# Step 4: Machine learning model
resume_df = pd.read_csv('resumes.csv') # Load resume data
X_train, X_test, y_train, y_test = train_test_split(resume_df['text'], resume_df['label'], test_size=0.2)

pipeline = Pipeline([
    ('vect', count_vect),
    ('tfidf', tfidf_transformer),
    ('clf', LogisticRegression())
])

parameters = {
    'vect__ngram_range': [(1, 1), (1, 2)],
    'tfidf__use_idf': (True, False),
    'clf__C': [1, 10]
}

grid_search = GridSearchCV(pipeline, parameters, cv=5)
grid_search.fit(X_train, y_train)

# Step 5: Evaluation
y_pred = grid_search.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
