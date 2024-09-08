import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage

# Load the Wholesale Customers dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00292/Wholesale%20customers%20data.csv"
df = pd.read_csv(url)

# Drop the 'Channel' and 'Region' columns for clustering, as they are categorical
X = df.drop(['Channel', 'Region'], axis=1)

# Standardize the data for better clustering performance
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 1. K-Means Clustering for Customer Segmentation
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans_labels = kmeans.fit_predict(X_scaled)

# Visualize K-Means Clusters using Annual spending in 'Milk' and 'Grocery'
plt.figure(figsize=(8, 6))
plt.scatter(X['Milk'], X['Grocery'], c=kmeans_labels, cmap='viridis', marker='o')
plt.title('K-Means Clustering on Wholesale Customers Data')
plt.xlabel('Annual Spending on Milk')
plt.ylabel('Annual Spending on Grocery')
plt.colorbar(label='Cluster')
plt.show()

# 2. Evaluating the Optimal Number of Clusters
inertia = []
silhouette_scores = []
cluster_range = range(2, 10)

for k in cluster_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans_labels = kmeans.fit_predict(X_scaled)
    
    # Compute inertia for Elbow Method
    inertia.append(kmeans.inertia_)
    
    # Compute Silhouette Score
    silhouette_avg = silhouette_score(X_scaled, kmeans_labels)
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
kmeans_labels = kmeans.fit_predict(X_scaled)

# 3. Cluster Analysis and Interpretation
df['Cluster'] = kmeans_labels

# Analyze the characteristics of each cluster
cluster_profiles = df.groupby('Cluster').mean()
print("Cluster Profiles:\n", cluster_profiles)

# Insights into customer segments based on different product categories
for i in range(optimal_k):
    cluster_size = df[df['Cluster'] == i].shape[0]
    avg_fresh = cluster_profiles.loc[i, 'Fresh']
    avg_milk = cluster_profiles.loc[i, 'Milk']
    avg_grocery = cluster_profiles.loc[i, 'Grocery']
    print(f"Cluster {i}: {cluster_size} customers, Avg Fresh: {avg_fresh:.2f}, Avg Milk: {avg_milk:.2f}, Avg Grocery: {avg_grocery:.2f}")

# 4. Hierarchical Clustering: Dendrogram and Cluster Formation
# Using Agglomerative Clustering
agg_cluster = AgglomerativeClustering(n_clusters=optimal_k)
agg_labels = agg_cluster.fit_predict(X_scaled)

# Compare the clusters with K-Means
df['Agg_Cluster'] = agg_labels

# Visualize clusters formed by Hierarchical Clustering
plt.figure(figsize=(8, 6))
plt.scatter(X['Milk'], X['Grocery'], c=agg_labels, cmap='plasma', marker='o')
plt.title('Hierarchical Clustering on Wholesale Customers Data')
plt.xlabel('Annual Spending on Milk')
plt.ylabel('Annual Spending on Grocery')
plt.colorbar(label='Cluster')
plt.show()

# Plot the Dendrogram for Hierarchical Clustering
linked = linkage(X_scaled, 'ward')

plt.figure(figsize=(10, 7))
dendrogram(linked, orientation='top', labels=agg_labels, distance_sort='descending', show_leaf_counts=True)
plt.title('Dendrogram for Hierarchical Clustering')
plt.xlabel('Sample Index')
plt.ylabel('Distance')
plt.show()

# 5. Comparison of Clustering Results
# Silhouette Scores Comparison
kmeans_silhouette = silhouette_score(X_scaled, kmeans_labels)
agg_silhouette = silhouette_score(X_scaled, agg_labels)

print(f"K-Means Silhouette Score: {kmeans_silhouette:.4f}")
print(f"Hierarchical Clustering Silhouette Score: {agg_silhouette:.4f}")

# PCA for better visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Visualize K-Means clusters in PCA-reduced space
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans_labels, cmap='viridis', marker='o')
plt.title('K-Means Clustering Visualized in PCA-Reduced Space')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(label='Cluster')
plt.show()

# Visualize Hierarchical Clustering in PCA-reduced space
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=agg_labels, cmap='plasma', marker='o')
plt.title('Hierarchical Clustering Visualized in PCA-Reduced Space')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(label='Cluster')
plt.show()

# Discussion of clustering results
print("\nDiscussion:")
print("- K-Means generally forms clusters with better cohesion and uniform size but is sensitive to the number of clusters chosen.")
print("- Hierarchical clustering does not assume any shape for the clusters and provides a full hierarchy of clusters, but it can be more computationally intensive for large datasets.")
