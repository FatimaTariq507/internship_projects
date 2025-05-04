import pandas as pd
from sklearn.cluster import KMeans

# Load data function
def load_data(path):
    data = pd.read_csv(path)
    # Encode Gender column: Male -> 1, Female -> 0
    if 'Gender' in data.columns:
        data['Gender'] = data['Gender'].map({'Male': 1, 'Female': 0})
    return data

# Clustering function
def perform_clustering(data, n_clusters=3):
    model = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = model.fit_predict(data)
    data['Cluster'] = cluster_labels
    return data, model

print("Model.py loaded successfully ✅")
