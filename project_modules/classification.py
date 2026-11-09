import os
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from project_modules.preprocess import *
from project_modules.dataset_merge import data_merge
from project_modules.rag_pipeline import *


def execute_classification(data_folder_path):
    ori_train_ds = pd.read_csv(os.path.join(data_folder_path, "train_claims.csv"))
    corpus = pd.read_csv(os.path.join(data_folder_path, "corpus.csv"))

    train_ds = data_merge(ori_train_ds,corpus)[['claim_text', 'text','label']]

    train_ds['text'] = train_ds['text'].fillna('')

    X = ( train_ds['claim_text'] + ' ' + train_ds['text'])
    y = train_ds['label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        shuffle=True,
        random_state=123
    )

    classifier_vectorizer = TfidfVectorizer(
        preprocessor=preprocess,
        tokenizer=tokenize,
        token_pattern=None,
        lowercase=False,
        ngram_range=(1, 5)
    )

    model = Pipeline([('vectorizer',classifier_vectorizer),
                        ('classifier',MultinomialNB(alpha=1))])

    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)


    print("Accuracy:", accuracy_score(y_test, y_pred))

    print(
        "Precision:",
        precision_score(
            y_test,
            y_pred,
            average="macro"
        )
    )

    print(
        "Recall:",
        recall_score(
            y_test,
            y_pred,
            average="macro"
        )
    )

    print(
        "Macro-F1:",
        f1_score(
            y_test,
            y_pred,
            average="macro"
        )
    )

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # start testing with testing dataset

    claims_test_ds_path = os.path.join(data_folder_path, "test_claims.csv")
    corpus_path = os.path.join(data_folder_path, "corpus.csv")




    test_ds = pd.read_csv(claims_test_ds_path)
    evidences_for_test_claims = get_evidences(claims_test_ds_path, corpus_path)
    highest_evidences, highest_evidences_ids = get_highest_evidences(evidences_for_test_claims)

    test_claim_ds = test_ds['claim_text']+ ' ' + highest_evidences

    unknown_test_result = model.predict_proba(test_claim_ds)

    confidences = []
    for confidence in unknown_test_result:
        confidences.append(confidence.max())

    predicted_labels = model.predict(test_claim_ds)

    for i,text in enumerate(highest_evidences):
        if text == '':
            highest_evidences[i] = 'NOT ENOUGH INFORMATION'

    result_dataset = pd.DataFrame({'claim_id':test_ds['claim_id'].tolist(),
                  'predicted_label': predicted_labels,
                  'retrieved_evidence_id':highest_evidences_ids,
                  'confidence':confidences,
                  'explanation':highest_evidences})



    return result_dataset

