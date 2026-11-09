import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from project_modules.preprocess import *
def evaluate_retrieval(train_claims_path, corpus_path, k=3):

    train_claims = pd.read_csv(train_claims_path)
    corpus = pd.read_csv(corpus_path)

    # Create TF-IDF vectors for corpus

    vectorizer = TfidfVectorizer(
        preprocessor=preprocess,
        tokenizer=tokenize,
        token_pattern=None,
        lowercase=False,
        ngram_range=(1, 1)
    )

    corpus_vector = vectorizer.fit_transform(corpus['text'])

    # Store evaluation scores

    precision_scores = []
    recall_scores = []
    reciprocal_ranks = []

    # Process every training claim

    for _, row in train_claims.iterrows():

        claim = row['claim_text']
        ground_truth = row['evidence_id']

        # Convert claim into TF-IDF
        claim_vector = vectorizer.transform([claim])

        # Calculate cosine similarity
        similarities = cosine_similarity(
            claim_vector,
            corpus_vector
        )[0]

        # Get Top-K evidence

        top_indices = similarities.argsort()[::-1][:k]

        retrieved_ids = []

        for index in top_indices:
            doc_id = corpus.iloc[index]['doc_id']
            retrieved_ids.append(doc_id)

        # Check ground-truth evidence

        relevant_count = retrieved_ids.count(ground_truth)

        # Precision@K

        precision = relevant_count / k

        # Recall@K

        recall = relevant_count / 1

        # Reciprocal Rank for MRR

        reciprocal_rank = 0

        for rank, doc_id in enumerate(retrieved_ids, start=1):

            if doc_id == ground_truth:
                reciprocal_rank = 1 / rank
                break

        # Store scores

        precision_scores.append(precision)
        recall_scores.append(recall)
        reciprocal_ranks.append(reciprocal_rank)


        print("\nClaim:", row['claim_id'])
        print("Ground truth:", ground_truth)
        print("Retrieved:", retrieved_ids)
        print("Precision@{}:".format(k), precision)
        print("Recall@{}:".format(k), recall)
        print("Reciprocal Rank:", reciprocal_rank)

    # Calculate final average scores

    precision_at_k = sum(precision_scores) / len(precision_scores)

    recall_at_k = sum(recall_scores) / len(recall_scores)

    mrr = sum(reciprocal_ranks) / len(reciprocal_ranks)

    # Print final evaluation

    print("\n==============================")
    print("Retrieval Evaluation")
    print("==============================")

    print("K =", k)
    print("Precision@{}:".format(k), precision_at_k)
    print("Recall@{}:".format(k), recall_at_k)
    print("MRR:", mrr)

    return precision_at_k, recall_at_k, mrr

