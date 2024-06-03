import pathlib
import json
import pickle
from collections import defaultdict
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from text_processing.TextProcessor import TextProcessor
from Tfidf.Tf_idf_Service import TfidfService
from Ranking.ranking_service import RankingService

# Instantiate services
tfidf_service = TfidfService()
textprocessor = TextProcessor()
rankingservice = RankingService()

# Load the saved vectorizer, TF-IDF matrix, and corpus
cwd = pathlib.Path().cwd()
clinical_tfidf_matrix_path = cwd / 'files' / 'clinicaltrials' / 'tfidf_matrix.pkl'
lifestyle_tfidf_matrix_path = cwd / 'files' / 'lifestyle' / 'tfidf_matrix.pkl'
clinical_corpus_key_path = cwd / 'files' / 'clinicaltrials' / 'corpuskey.json'
lifestyle_corpus_key_path = cwd / 'files' / 'lifestyle' / 'corpuskey.json'
lifestyle_folder = cwd / "files" / "lifestyle/"
clinical_folder = cwd / "files" / "clinicaltrials/"
tfidf_service.preload(lifestyle_folder, clinical_folder)

with open(clinical_corpus_key_path, 'r') as f:
    clinical_corpus_keys = json.load(f)

with open(lifestyle_corpus_key_path, 'r') as f:
    lifestyle_corpus_keys = json.load(f)

# Load K-means model and inverted index
kmeans_model_path = cwd / "files" / 'clinicaltrials' / "kmeans_model.pkl"
inverted_index_path = cwd / "files" / 'clinicaltrials' / "inverted_index_kmeans.json"

with open(kmeans_model_path, 'rb') as file:
    kmeans_model = pickle.load(file)

with open(inverted_index_path, 'r') as file:
    inverted_index = json.load(file)

def docsIdsSearchWithKmeans(query_text, corpus_keys, dataset, kmeans_model, inverted_index):
    # Ensure the query is processed correctly
    processed_query = textprocessor.process_text(query_text)
    
    # Vectorize the query
    query_vector = tfidf_service.vectorize_query_evaluation(processed_query, dataset)
    
    # Predict the cluster for the query vector
    cluster_label = kmeans_model.predict(query_vector)[0]
    
    # Retrieve documents in the predicted cluster
    cluster_docs = inverted_index.get(str(cluster_label), [])
    if not cluster_docs:
        return []  # No documents found in the cluster
    
    # Calculate similarity scores for documents in the selected cluster
    cluster_docs_vectors = tfidf_service.get_docs_vectors(cluster_docs, dataset)
    similarity_scores = cosine_similarity(query_vector, cluster_docs_vectors).flatten()
    
    # Rank documents based on similarity scores
    ranked_docs = rankingservice.rank_and_sort(similarity_scores)
    
    # Return the document keys
    return [corpus_keys[doc_idx] for doc_idx, _ in ranked_docs]

def load_qrel_clinical(jsonl_file_path):
    qrel_dict = defaultdict(dict)
    real_relevant = defaultdict(list)
    with open(jsonl_file_path, 'r', encoding='utf-8') as file:
        for line in file:
            entry = json.loads(line.strip())
            query_id = entry["query_id"]
            docs = entry["docs"]
            for doc in docs:
                doc_id = doc["doc_id"]
                relevance = doc["relevance"]
                qrel_dict[query_id][doc_id] = relevance
                if relevance > 0:
                    real_relevant[query_id].append(doc_id)
    return qrel_dict, real_relevant

clinical_qrel_path = cwd / "files" / 'clinicaltrials' / "qrel.jsonl"
clinical_qrel_dict, real_relevant_clinical = load_qrel_clinical(clinical_qrel_path)

retrieved_docs_clinical_kmeans = {}
print("Retrieving documents for each clinical query using K-means...")
dataset = "clinicaltrials"
with open(clinical_qrel_path, 'r', encoding='utf-8') as file:
    for line in file:
        entry = json.loads(line.strip())
        query_id = entry["query_id"]
        query_text = entry["text"]
        ret_docs = docsIdsSearchWithKmeans(query_text, clinical_corpus_keys, dataset, kmeans_model, inverted_index)
        retrieved_docs_clinical_kmeans[query_id] = ret_docs
        print(query_id)
print("Clinical document retrieval with K-means complete.")

class EvaluationMetrics:
    def __init__(self, true_data, predictions):
        self.true_data = {str(k): list(map(str, v)) for k, v in true_data.items()}
        self.predictions = {str(k): list(map(str, v)) for k, v in predictions.items()}

    def calculate_recall(self, true_pids, pred_indices):
        true_set = set(true_pids)
        pred_set = set(pred_indices)
        if len(true_set) == 0:
            return 0
        return len(true_set & pred_set) / len(true_set)

    def calculate_precision_at_k(self, true_pids, pred_indices, k):
        true_set = set(true_pids)
        pred_set = set(pred_indices[:k])
        if len(pred_set) == 0:
            return 0
        return len(true_set & pred_set) / k

    def average_precision(self, true_pids, pred_indices):
        relevant = 0
        sum_precisions = 0
        for i, pred in enumerate(pred_indices):
            if pred in true_pids:
                relevant += 1
                sum_precisions += relevant / (i + 1)
        if relevant == 0:
            return 0
        return sum_precisions / len(true_pids)

    def mean_reciprocal_rank(self):
        mrr_sum = 0
        for query_id, true_ids in self.true_data.items():
            pred_ids = self.predictions.get(query_id, [])
            for rank, doc_id in enumerate(pred_ids, start=1):
                if doc_id in true_ids:
                    mrr_sum += 1 / rank
                    break
        return mrr_sum / len(self.true_data)

    def calculate_metrics(self):
        recalls = []
        precisions_k = []
        aps = []

        for query_id, true_ids in self.true_data.items():
            pred_ids = self.predictions.get(query_id, [])
            recalls.append(self.calculate_recall(true_ids, pred_ids))
            precisions_k.append(self.calculate_precision_at_k(true_ids, pred_ids, 10))
            aps.append(self.average_precision(true_ids, pred_ids))

        mean_recall = sum(recalls) / len(recalls)
        mean_precision_at_k = sum(precisions_k) / len(precisions_k)
        mean_ap = sum(aps) / len(aps)
        mrr = self.mean_reciprocal_rank()

        print(f"Mean Recall: {mean_recall}")
        print(f"Precision@10: {mean_precision_at_k}")
        print(f"Mean Average Precision: {mean_ap}")
        print(f"Mean Reciprocal Rank: {mrr}")

        return mean_recall, mean_precision_at_k, mean_ap, mrr

# Evaluate the performance using K-means-based retrieval
print("Evaluating clinical trial metrics with K-means...")
evaluation_clinical_kmeans = EvaluationMetrics(real_relevant_clinical, retrieved_docs_clinical_kmeans)
evaluation_clinical_kmeans.calculate_metrics()
