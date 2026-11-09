# NLP Claim Verification & Evidence Retrieval

## Project Overview

NLP Claim Verification & Evidence Retrieval is a Natural Language Processing (NLP) project designed to automatically verify factual claims using a collection of reference documents.

Given a natural-language claim and a document corpus, the system retrieves relevant evidence passages and classifies the claim into one of three categories:

- **SUPPORTS:** The retrieved evidence provides information supporting the claim.
- **REFUTES:** The retrieved evidence provides information contradicting the claim.
- **NOT ENOUGH INFO:** The available documents do not contain sufficient evidence to verify or refute the claim.

This project explores two complementary NLP approaches: traditional statistical text classification and semantic retrieval with Retrieval-Augmented Generation (RAG).

## Key Features

### 1. Text Preprocessing
- Lowercase text normalization.
- Punctuation removal and text cleaning.
- Tokenization and stopword removal.
- Preparation of text data for feature extraction and classification.

### 2. TF-IDF Feature Extraction
- Converts text documents into numerical feature representations.
- Uses TF-IDF (Term Frequency–Inverse Document Frequency) to extract important textual features.
- Generates sparse feature matrices for machine learning classification.

### 3. Naïve Bayes Classification
- Implements a probabilistic text classification approach using Naïve Bayes.
- Uses Multinomial or Complement Naïve Bayes with Laplace smoothing.
- Classifies claims into SUPPORTS, REFUTES, and NOT ENOUGH INFO.

### 4. Evidence Retrieval
- Represents documents in a vector space.
- Uses cosine similarity to measure textual similarity between claims and reference passages.
- Retrieves the Top-K most relevant evidence passages.

### 5. RAG-Based Claim Verification
- Combines claim information with retrieved evidence.
- Uses evidence-grounded reasoning to determine the claim stance.
- Generates concise explanations based on the retrieved evidence.

### 6. Model Evaluation
Evaluates classification and retrieval performance using:

**Classification Metrics**
- Accuracy
- Precision
- Recall
- Macro F1-Score
- Micro F1-Score
- Confusion Matrix

**Information Retrieval Metrics**
- Precision@K
- Recall@K
- Mean Reciprocal Rank (MRR)

## Technologies Used

| Category | Technologies |
|---|---|
| Programming Language | Python |
| NLP | Natural Language Processing |
| Text Feature Extraction | TF-IDF |
| Machine Learning | Naïve Bayes |
| Retrieval | Cosine Similarity |
| RAG | Retrieval-Augmented Generation |
| Data Processing | Pandas, NumPy |
| Evaluation | Scikit-learn |

## NLP Pipeline

The project explores the following processing pipeline:

**Traditional Statistical NLP:**

1. Text Preprocessing
2. TF-IDF Feature Extraction
3. Naïve Bayes Classification
4. Multi-class Stance Prediction
5. Classification Performance Evaluation

**Semantic Retrieval & RAG:**

1. Document Vector Representation
2. Cosine Similarity Calculation
3. Top-K Evidence Retrieval
4. Evidence-Grounded Claim Verification
5. Explanation Generation
6. Retrieval Performance Evaluation

## Author

**Min Kaung Khant**

- GitHub: [MinKaungKhantOne](https://github.com/MinKaungKhantOne)

