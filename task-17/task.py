import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn import *

# Load the Mall Customers dataset

df = pd.read_csv('task-17/datasets/mall_customers.csv')

# We will use 'Annual Income (k$)' and 'Spending Score (1-100)' for clustering
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

# 1. Implementing K-Means Clustering on Customer Segments
# Apply K-Means Clustering with 3 clusters initially
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans_labels = kmeans.fit_predict(X)

# Visualize K-Means Clusters
plt.figure(figsize=(8, 6))
plt.scatter(X['Annual Income (k$)'], X['Spending Score (1-100)'], c=kmeans_labels, cmap='viridis', marker='o')
plt.title('K-Means Clustering of Mall Customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.colorbar(label='Cluster')
plt.show()

# 2. Optimal Number of Clusters: Elbow Method and Silhouette Score
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

# Based on the plots, choose the optimal number of clusters
optimal_k = 5  # Example: based on elbow and silhouette score, assume 5 is optimal
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
kmeans_labels = kmeans.fit_predict(X)

# 3. Cluster Profiling and Insights
df['Cluster'] = kmeans_labels

# Analyze the characteristics of each cluster
cluster_profiles = df.groupby('Cluster').mean()
print("Cluster Profiles:\n", cluster_profiles)

# Insights into customer segments based on Annual Income and Spending Score
for i in range(optimal_k):
    cluster_size = df[df['Cluster'] == i].shape[0]
    avg_income = cluster_profiles.loc[i, 'Annual Income (k$)']
    avg_spending = cluster_profiles.loc[i, 'Spending Score (1-100)']
    print(f"Cluster {i}: {cluster_size} customers, Avg Income: {avg_income:.2f}k$, Avg Spending Score: {avg_spending:.2f}")

# 4. Hierarchical Clustering for Customer Segmentation
# Using Agglomerative Clustering
agg_cluster = AgglomerativeClustering(n_clusters=optimal_k)
agg_labels = agg_cluster.fit_predict(X)

# Compare the clusters with K-Means
df['Agg_Cluster'] = agg_labels

# Visualize clusters formed by Hierarchical Clustering
plt.figure(figsize=(8, 6))
plt.scatter(X['Annual Income (k$)'], X['Spending Score (1-100)'], c=agg_labels, cmap='plasma', marker='o')
plt.title('Hierarchical Clustering of Mall Customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.colorbar(label='Cluster')
plt.show()

# Plot the Dendrogram for Hierarchical Clustering
linked = linkage(X, 'ward')

plt.figure(figsize=(10, 7))
dendrogram(linked, orientation='top', labels=agg_labels, distance_sort='descending', show_leaf_counts=True)
plt.title('Dendrogram for Hierarchical Clustering')
plt.xlabel('Sample Index')
plt.ylabel('Distance')
plt.show()

# 5. Visualizing Clusters with PCA
# Standardize the data for PCA
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA to reduce to 2 dimensions
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Visualize the K-Means clusters in PCA-reduced space
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans_labels, cmap='viridis', marker='o')
plt.title('K-Means Clustering Visualized in PCA-Reduced Space')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(label='Cluster')
plt.show()

# Visualize the Agglomerative clusters in PCA-reduced space
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=agg_labels, cmap='plasma', marker='o')
plt.title('Hierarchical Clustering Visualized in PCA-Reduced Space')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(label='Cluster')
plt.show()

