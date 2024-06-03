import json
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pathlib
from scipy.sparse import vstack
class TfidfService:
    def __init__(self):
        self.lifestyle_vectorizer = None
        self.lifestyle_tfidf_matrix = None
        self.clinical_vectorizer = None
        self.clinical_tfidf_matrix = None

       
        

    def fit_transform_documents(self, documents):
        self.vectorizer = TfidfVectorizer()
        tfidf_matrix = self.vectorizer.fit_transform(documents)
        return tfidf_matrix

    def save_tfidf_matrix(self, tfidf_matrix, file_path):
        with open(file_path, "wb") as f:
            pickle.dump(tfidf_matrix, f)

    def save_vectorizer(self, file_path):
        with open(file_path, "wb") as f:
            pickle.dump(self.vectorizer, f)

    def process_json_file(self, input_file_path, tfidf_matrix_path, vectorizer_path):
        with open(input_file_path, "r") as f:
            corpus = json.load(f)
        documents = [str(doc_text) for doc_id, doc_text in corpus.items()]
        tfidf_matrix = self.fit_transform_documents(documents)
        self.save_tfidf_matrix(tfidf_matrix, tfidf_matrix_path)
        self.save_vectorizer(vectorizer_path)

    def load_vectorizer(self, vectorizer_path):
        with open(vectorizer_path, "rb") as f:
            return pickle.load(f)

    def load_tfidf_matrix(self, tfidf_matrix_path):
        with open(tfidf_matrix_path, "rb") as f:
            return pickle.load(f)
    def preload(self, lifestyle_folder, clinical_folder):
        try:
            self.lifestyle_vectorizer = self.load_vectorizer(lifestyle_folder / "vectorizer.pkl")
            self.lifestyle_tfidf_matrix = self.load_tfidf_matrix(lifestyle_folder / "tfidf_matrix.pkl")
            self.clinical_vectorizer = self.load_vectorizer(clinical_folder / "vectorizer.pkl")
            self.clinical_tfidf_matrix = self.load_tfidf_matrix(clinical_folder / "tfidf_matrix.pkl")
            print("Preloading successful")
        except Exception as e:
            print(f"Error during preloading: {e}")

    def vectorize_query(self, query_tokens, dataset):
        if dataset == "lifestyle":
            vectorizer = self.lifestyle_vectorizer
        elif dataset == "clinicaltrials":
            vectorizer = self.clinical_vectorizer
        else:
            raise ValueError(f"Dataset {dataset} not found")
        
        processed_query_string = ' '.join(query_tokens)
        query_vector = vectorizer.transform([processed_query_string])
        return {
            "shape": query_vector.shape,
            "data": query_vector.data.tolist(),
            "indices": query_vector.indices.tolist(),
            "indptr": query_vector.indptr.tolist()
        }
    
    def vectorize_query_evaluation(self, query_tokens, dataset):
        if dataset == "lifestyle":
            vectorizer = self.lifestyle_vectorizer
        elif dataset == "clinicaltrials":
            vectorizer = self.clinical_vectorizer
        else:
            raise ValueError(f"Dataset {dataset} not found")
        
        if vectorizer is None:
            raise ValueError(f"Vectorizer for dataset {dataset} is not loaded")
        processed_query_string = ' '.join(query_tokens)
        query_vector = vectorizer.transform([processed_query_string])
        return query_vector
    
    def calculate_similarity(self, query_vector, dataset):
        if dataset == "lifestyle":
            tfidf_matrix = self.lifestyle_tfidf_matrix
        elif dataset == "clinicaltrials":
            tfidf_matrix = self.clinical_tfidf_matrix
        else:
            raise ValueError(f"Dataset {dataset} not found")
        similarity_matrix = cosine_similarity(query_vector, tfidf_matrix)
        return similarity_matrix.flatten()
    
    def calculate_similarity_api(self, query_vector, dataset):
            if dataset == "lifestyle":
                tfidf_matrix = self.lifestyle_tfidf_matrix
            elif dataset == "clinicaltrials":
                tfidf_matrix = self.clinical_tfidf_matrix
            else:
                raise ValueError(f"Dataset {dataset} not found")
            similarity_matrix = cosine_similarity(query_vector, tfidf_matrix)
            return similarity_matrix.flatten().tolist()

    def get_docs_vectors(self, doc_indices, dataset):
        if dataset == "lifestyle":
            return vstack([self.lifestyle_tfidf_matrix[idx] for idx in doc_indices])
        elif dataset == "clinicaltrials":
            return vstack([self.clinical_tfidf_matrix[idx] for idx in doc_indices])
        else:
            raise ValueError(f"Dataset {dataset} not found")

