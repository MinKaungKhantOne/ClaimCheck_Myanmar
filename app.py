"""
MMDT Myanmar NLP (MNLP) Project: Myanmar Text Claim Verification & Evidence Retrieval - Main Entry Point

Instructions for Students:
1. You have complete flexibility in how you design your project folder structure, classes, and helper modules.
2. Build your custom Myanmar reference corpus (corpus.csv) and curate at least 50 Myanmar claims (claims.csv / train_claims.csv / test_claims.csv).
3. Implement your Myanmar NLP verification and retrieval pipeline inside or called from `studentverifier()`.
4. Your verifier should read data dynamically from `data_dir` (default: 'data') using UTF-8 encoding.
   - If datasets in `data/` are updated or replaced, your solution should still work dynamically.
5. Your system must generate the final prediction file and save it as `<studentid>_predictions.csv` inside `output_dir` (default: 'data').
6. The function `studentverifier()` must return the path to the created CSV file.
7. In your report and video presentation, explain how your project directory, Myanmar syllable/word tokenization, TF-IDF vectorizer, Naïve Bayes classifier, and semantic retrieval/RAG pipelines are built.
"""

import os
import re
import csv
from project_modules.preprocess import *
from project_modules.classification import *
from project_modules.evaluation_rag import *

# TODO: Import your custom Myanmar NLP modules, tokenizers, vectorizers, classifiers, or RAG modules here.
# Example:
# from mnlp_engine.preprocessor import clean_myanmar_text, syllable_tokenizer
# from mnlp_engine.naive_bayes import MultinomialNaiveBayesClassifier
# from mnlp_engine.retriever import MyanmarSemanticRetriever
# from mnlp_engine.rag_verifier import MyanmarRAGClaimVerifier


def get_student_id():
    """
    Helper function to extract your MMDT ID from report.md.
    Alternatively, you may specify your student ID directly.
    """
    report_path = os.path.join(os.path.dirname(__file__), 'report.md')
    if os.path.exists(report_path):
        try:
            with open(report_path, 'r', encoding='utf-8') as f:
                content = f.read()
                match = re.search(r'-\s*\*\*MMDT ID:\*\*\s*(.*)', content)
                if match:
                    val = match.group(1).strip()
                    if val and not (val.startswith('[') and val.endswith(']')):
                        return re.sub(r'[^a-zA-Z0-9_-]', '_', val)
        except Exception:
            pass
    return "student"


def studentverifier(data_dir="data", output_dir="data", student_id=None):
    """
    Main entry point for Myanmar text claim verification and evidence retrieval.

    Pipeline Steps:
    1. Myanmar Text Preprocessing & Syllable/Word Tokenization (UTF-8)
    2. Feature Extraction (TF-IDF / Syllable N-grams / Dense Embeddings)
    3. Evidence Retrieval (Vector / Lexical similarity search from corpus.csv)
    4. Stance / Claim Classification (SUPPORTS / REFUTES / NOT ENOUGH INFO)
    5. Evidence-Grounded Explanation / RAG Synthesis in Myanmar context

    Args:
        data_dir (str): Directory containing input CSV files (corpus.csv, claims.csv, train_claims.csv, test_claims.csv).
        output_dir (str): Directory where '<studentid>_predictions.csv' will be saved.
        student_id (str, optional): The student ID to use in the filename. Defaults to ID in report.md.

    Returns:
        str: Path to the generated '<studentid>_predictions.csv' file.
    """
    if student_id is None:
        student_id = get_student_id()

    os.makedirs(output_dir, exist_ok=True)
    output_filename = f"{student_id}_predictions.csv"
    output_path = os.path.join(output_dir, output_filename)

    # =========================================================================
    # TODO: Implement your Myanmar NLP Claim Verification & Retrieval Pipeline below.
    #
    # 1. Load corpus.csv, train_claims.csv, and test_claims.csv dynamically with utf-8 encoding.
    # 2. Preprocess Myanmar text (Unicode normalization, syllable segmentation / regex tokenization, stopword removal).
    # 3. Build TF-IDF / Vector space representations for Myanmar documents and claims.
    # 4. Perform Top-K evidence retrieval for each claim against corpus passages.
    # 5. Classify claims using Naïve Bayes baseline and/or RAG verification.
    # 6. Save predictions to `output_path`.
    #
    # Expected CSV columns:
    # claim_id,predicted_label,retrieved_evidence_id,confidence,explanation
    #
    # Labels must be one of: 'SUPPORTS', 'REFUTES', 'NOT ENOUGH INFO'
    # =========================================================================

    base_dir = os.path.dirname(os.path.abspath(__file__))

    data_folder_path = os.path.join(base_dir, data_dir)

    train_ds_path = os.path.join(data_folder_path, "train_claims.csv")
    corpus_path = os.path.join(data_folder_path, "corpus.csv")

    result_dataset = execute_classification(data_folder_path)
    evaluate_retrieval(train_ds_path, corpus_path)

    result_dataset.to_csv(output_path)

    return output_path


if __name__ == "__main__":
    generated_file = studentverifier()
    print(f"Myanmar claim verification completed. Output file: {generated_file}")
