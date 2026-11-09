import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from project_modules.preprocess import *


def get_evidences(test_claims_path, corpus_path):
    ori_test_ds = pd.read_csv(test_claims_path)
    corpus = pd.read_csv(corpus_path)

    test_ds = ori_test_ds['claim_text']

    test_vectorizer = TfidfVectorizer(
        preprocessor=preprocess,
        tokenizer=tokenize,
        token_pattern=None,
        lowercase=False,
        ngram_range=(1,1)
    )

    corpus_vector = test_vectorizer.fit_transform(corpus['text'])
    claims_vector = test_vectorizer.transform(test_ds)

    similarities = cosine_similarity(
        claims_vector,
        corpus_vector
    )

    # top_indices = similarities[0].argsort()[::-1][:k]
    #
    # for index in top_indices:
    #     print("Index:", index)
    #     print("Doc ID:", corpus.iloc[index]["doc_id"])
    #     print("Score:", similarities[0][index])
    #     print("Text:", corpus.iloc[index]["text"])
    #     print()

    k = 3
    if k == 3:

        all_evidences = []

        for each_simil in similarities:
            top_indices = each_simil.argsort()[::-1][:k]
            evidences = []
            evidence_info = {}

            for rank, index in enumerate(top_indices):
                if rank == 0:
                    evidence_info['highest'] = {'index': index, 'doc_id': corpus.iloc[index]["doc_id"],
                                                'score':each_simil[index],'text': corpus.iloc[index]["text"]}
                elif rank == 1:
                    evidence_info['middle'] = {'index': index, 'doc_id': corpus.iloc[index]["doc_id"],
                                                'score': each_simil[index], 'text': corpus.iloc[index]["text"]}
                else:
                    evidence_info['lowest'] = {'index': index, 'doc_id': corpus.iloc[index]["doc_id"],
                                                'score': each_simil[index], 'text': corpus.iloc[index]["text"]}

            evidences.append(evidence_info)
            all_evidences.extend(evidences)

        return all_evidences


def get_highest_evidences(all_evidences):
    evidences_text = []
    evidences_ids = []
    threshold = 0.50
    for each_claim_evidence in all_evidences:
        evidences_ids.append(each_claim_evidence['highest']['doc_id'])
        if each_claim_evidence['highest'] and each_claim_evidence['highest']['score'] > threshold:
            evidences_text.append(each_claim_evidence['highest']['text'])
        else:
            evidences_text.append('')
    return evidences_text,evidences_ids
#
#
# for i in get_highest_evidences(all_evidences):
#     print(i)

