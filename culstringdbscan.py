
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from pathlib import Path
import pickle
import json
from collections import defaultdict
from Tfidf.Tf_idf_Service import TfidfService

# Set the current working directory
cwd = Path().cwd()

# Define the paths to the TF-IDF matrices
lifestyle_tfidf_matrix_path = cwd / 'files' / 'lifestyle' / 'tfidf_matrix.pkl'

# Initialize the TfidfService and load the TF-IDF matrix for lifestyle documents
tfidfservice = TfidfService()
tfidf_matrix = tfidfservice.load_tfidf_matrix(lifestyle_tfidf_matrix_path)

# Normalize the TF-IDF matrix
scaler = StandardScaler(with_mean=False)
tfidf_matrix_normalized = scaler.fit_transform(tfidf_matrix)

# Apply DBSCAN
eps = 0.5  # maximum distance between two samples for them to be considered as in the same neighborhood
min_samples = 5  # number of samples in a neighborhood for a point to be considered as a core point

dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
labels = dbscan.fit_predict(tfidf_matrix_normalized)

# Evaluate the clustering performance
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = list(labels).count(-1)

print(f'Estimated number of clusters: {n_clusters}')
print(f'Estimated number of noise points: {n_noise}')

# Silhouette score is only meaningful if there are more than 1 clusters
if n_clusters > 1:
    silhouette_avg = silhouette_score(tfidf_matrix_normalized, labels)
    print(f'Silhouette Score: {silhouette_avg}')
else:
    print('Silhouette Score: Not applicable (only one cluster)')

# Save the DBSCAN model
dbscan_model_path = cwd / 'files' / 'lifestyle' / "dbscan_model.pkl"
with open(dbscan_model_path, 'wb') as f:
    pickle.dump(dbscan, f)

print("DBSCAN model saved.")

# Create an inverted index for clusters
inverted_index = defaultdict(list)
for idx, label in enumerate(labels):
    inverted_index[str(label)].append(idx)

# Save the inverted index
inverted_index_path = cwd / 'files' / 'lifestyle' / "inverted_index_dbscan.json"
with open(inverted_index_path, 'w') as f:
    json.dump(inverted_index, f)

print("Inverted index created.")

# Check for NaN, Inf values and ensure data is finite
if not np.all(np.isfinite(tfidf_matrix_normalized)):
    print("Data contains non-finite values. Cleaning data...")
    tfidf_matrix_normalized = np.nan_to_num(tfidf_matrix_normalized, nan=0.0, posinf=1e5, neginf=-1e5)

# Reduce dimensionality for visualization using PCA
pca = PCA(n_components=2, random_state=42)
try:
    reduced_data = pca.fit_transform(tfidf_matrix_normalized)
except Exception as e:
    print(f"Error during PCA transformation: {e}")
    exit()

# Visualize clusters using PCA
plt.figure(figsize=(10, 6))
unique_labels = set(labels)
colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]

for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 1]

    class_member_mask = (labels == k)

    xy = reduced_data[class_member_mask]
    plt.scatter(xy[:, 0], xy[:, 1], marker='o', color=tuple(col), edgecolor='k', s=50, label=f"Cluster {k}")

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('DBSCAN Clustering (PCA-reduced data)')
plt.legend()
plt.show()
