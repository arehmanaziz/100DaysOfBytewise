import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target
species = iris.target_names

# 1. Implementing K-Means Clustering
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans_labels = kmeans.fit_predict(X)

# Scatter plot of two features: Sepal Length and Sepal Width
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=kmeans_labels, cmap='viridis', marker='o')
plt.title('K-Means Clustering on Iris Dataset (Sepal Length vs Sepal Width)')
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.colorbar(label='Cluster')
plt.show()

# Compare with actual species labels
print("Cluster centers:\n", kmeans.cluster_centers_)
print("KMeans vs True Labels:\n", pd.crosstab(kmeans_labels, y))

# 2. Choosing the Optimal Number of Clusters (Elbow Method and Silhouette Score)
inertia = []
silhouette_scores = []
cluster_range = range(2, 10)

for k in cluster_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans_labels = kmeans.fit_predict(X)
    
    # Compute inertia for Elbow Method
    inertia.append(kmeans.inertia_)
    
    # Compute Silhouette Score
    silhouette_avg = silhouette_score(X, kmeans_labels)
    silhouette_scores.append(silhouette_avg)

# Plot Elbow Method
plt.figure(figsize=(8, 6))
plt.plot(cluster_range, inertia, marker='o')
plt.title('Elbow Method for Optimal Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.show()

# Plot Silhouette Scores
plt.figure(figsize=(8, 6))
plt.plot(cluster_range, silhouette_scores, marker='o')
plt.title('Silhouette Scores for Different Cluster Counts')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.show()

# 3. Cluster Visualization with PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

kmeans = KMeans(n_clusters=3, random_state=42)
kmeans_labels = kmeans.fit_predict(X)

# Scatter plot of PCA-reduced data
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans_labels, cmap='viridis', marker='o')
plt.title('K-Means Clustering on PCA-Reduced Iris Data')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(label='Cluster')
plt.show()

# 4. Hierarchical Clustering: Dendrogram
linked = linkage(X, 'ward')

plt.figure(figsize=(10, 7))
dendrogram(linked, orientation='top', labels=y, distance_sort='descending', show_leaf_counts=True)
plt.title('Dendrogram for Hierarchical Clustering')
plt.xlabel('Sample Index')
plt.ylabel('Distance')
plt.show()

# 5. Comparing Clustering Algorithms: K-Means vs Agglomerative Clustering
# Agglomerative Clustering
agg_clustering = AgglomerativeClustering(n_clusters=3)
agg_labels = agg_clustering.fit_predict(X)

# Compare performance
kmeans_silhouette = silhouette_score(X, kmeans_labels)
agg_silhouette = silhouette_score(X, agg_labels)

print(f"K-Means Silhouette Score: {kmeans_silhouette:.4f}")
print(f"Agglomerative Clustering Silhouette Score: {agg_silhouette:.4f}")

# Scatter plot for comparison (PCA reduced)
fig, ax = plt.subplots(1, 2, figsize=(16, 6))

# K-Means Clustering
ax[0].scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans_labels, cmap='viridis', marker='o')
ax[0].set_title('K-Means Clustering (PCA Reduced)')
ax[0].set_xlabel('Principal Component 1')
ax[0].set_ylabel('Principal Component 2')

# Agglomerative Clustering
ax[1].scatter(X_pca[:, 0], X_pca[:, 1], c=agg_labels, cmap='viridis', marker='o')
ax[1].set_title('Agglomerative Clustering (PCA Reduced)')
ax[1].set_xlabel('Principal Component 1')
ax[1].set_ylabel('Principal Component 2')

plt.show()

# Discuss the strengths and weaknesses of each approach:
print("\nDiscussion:")
print("- K-Means is faster and works well when clusters are roughly spherical and similar in size. However, it can struggle with clusters of different sizes and densities.")
print("- Agglomerative Clustering does not assume any particular shape for the clusters and is more flexible, but it can be computationally expensive and sensitive to outliers.")
