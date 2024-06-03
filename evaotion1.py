import pathlib
import json
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from collections import defaultdict
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




def docsIdsSearch(query_text, corpus_keys, dataset):
    processed_query = textprocessor.process_text(query_text)
    
    query_vector = tfidf_service.vectorize_query_evaluation(processed_query, dataset)
    
    similarity_scores = tfidf_service.calculate_similarity(query_vector, dataset)
    
    sorted_ranks = rankingservice.rank_and_sort(similarity_scores)
    return [corpus_keys[doc_idx] for doc_idx, _ in sorted_ranks]

# Load the qrel JSONL file for clinical trials
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

# # Load the qrel JSONL file for lifestyle
# def load_qrel_lifestyle(jsonl_file_path):
#     qrel_dict = defaultdict(list)
#     with open(jsonl_file_path, 'r', encoding='utf-8') as file:
#         for line in file:
#             entry = json.loads(line.strip())
#             query_id = entry["qid"]
#             answer_pids = entry["answer_pids"]
#             qrel_dict[query_id] = answer_pids
#     return qrel_dict

# Paths to QREL files
clinical_qrel_path = cwd / "files" / 'clinicaltrials' / "qrel.jsonl"
# lifestyle_qrel_path = cwd / "files" / 'lifestyle' / "qas.forum.jsonl"

# Load QREL data
clinical_qrel_dict, real_relevant_clinical = load_qrel_clinical(clinical_qrel_path)
# lifestyle_qrel_dict = load_qrel_lifestyle(lifestyle_qrel_path)

# Retrieve documents for each query using the search service
retrieved_docs_clinical = {}
print("Retrieving documents for each clinical query...")
dataset = "clinicaltrials"
with open(clinical_qrel_path, 'r', encoding='utf-8') as file:
    for line in file:
        entry = json.loads(line.strip())
        query_id = entry["query_id"]
        query_text = entry["text"]
        ret_docs = docsIdsSearch(query_text, clinical_corpus_keys, dataset)
        retrieved_docs_clinical[query_id] = ret_docs
        print(query_id)
print("Clinical document retrieval complete.")

# # Add function to retrieve lifestyle documents
# retrieved_docs_lifestyle = {}
# print("Retrieving documents for each lifestyle query...")
# dataset = "lifestyle"
# with open(lifestyle_qrel_path, 'r', encoding='utf-8') as file:
#     for line in file:
#         entry = json.loads(line.strip())
#         query_id = entry["qid"]
#         query_text = entry["query"]
#         ret_docs = docsIdsSearch(query_text, lifestyle_corpus_keys, dataset)
#         retrieved_docs_lifestyle[query_id] = ret_docs
#         print(query_id)
# print("Lifestyle document retrieval complete.")

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
        print(f"Mean Recall: {mean_recall}")
        print(f"Precision@10: {mean_precision_at_k}")
        print(f"Mean Average Precision: {mean_ap}")

        return mean_recall, mean_precision_at_k, mean_ap
# retrieved_docs_lifestyle_path = cwd / "files" / 'lifestyle' / "retrieved_docs_lifestyle.pkl"

# import json

# # Save the dictionary to a JSON file
# with open(retrieved_docs_lifestyle_path, 'w') as f:
#     json.dump(retrieved_docs_lifestyle, f)

print("retrieved_docs_lifestyle saved as JSON.")

# Calculate and print the metrics for clinical trials
print("Evaluating clinical trial metrics...")
evaluation_clinical = EvaluationMetrics(real_relevant_clinical, retrieved_docs_clinical)
evaluation_clinical.calculate_metrics()

# # Calculate and print the metrics for lifestyle
# print("Evaluating lifestyle metrics...")
# evaluation_lifestyle = EvaluationMetrics(lifestyle_qrel_dict, retrieved_docs_lifestyle)
# evaluation_lifestyle.calculate_metrics()
