import pandas as pd
import numpy as np
from collections import Counter
from sklearn.cluster import KMeans

class GSAILogic:
    def __init__(self, file_path):
        # Load and prepare data
        self.df = pd.read_csv(file_path, header=None)
        
        # Pull coordinates for visualization (source bytes and dest bytes)
        self.points = np.log1p(self.df.iloc[:500, [4, 5]].values) 
        
        # Secure the ground-truth classification labels from column index 41
        self.labels = self.df.iloc[:500, 41].values
        
        # Normalize logged data to 0.15 - 0.85 to give padding from the canvas walls
        p_min = self.points.min(axis=0)
        p_max = self.points.max(axis=0)
        
        range_denom = np.where((p_max - p_min) == 0, 1, p_max - p_min)
        self.points = 0.15 + 0.7 * (self.points - p_min) / range_denom
        
        self.centroids = []
        self.clusters = []

    def step_calculate_clusters(self):
        """Phase 1: Assign each point to the nearest centroid"""
        distances = np.linalg.norm(self.points[:, np.newaxis] - self.centroids, axis=2)
        self.clusters = np.argmin(distances, axis=1)
        return self.clusters

    def step_move_centroids(self):
        """Phase 2: Move centroids to the mean of their clusters and check for convergence"""
        old_centroids = self.centroids.copy()
        
        new_centroids = np.array([self.points[self.clusters == i].mean(axis=0) 
                                 if len(self.points[self.clusters == i]) > 0 else self.centroids[i]
                                 for i in range(len(self.centroids))])
        self.centroids = new_centroids
        
        has_converged = np.allclose(old_centroids, new_centroids, atol=1e-6)
        return has_converged

    def get_cluster_labels(self):
        """Analyzes the network logs inside each cluster and returns the majority security tag"""
        cluster_tags = []
        if len(self.clusters) == 0:
            return ["Unassigned"] * len(self.centroids)
            
        for i in range(len(self.centroids)):
            points_in_cluster = self.labels[self.clusters == i]
            
            if len(points_in_cluster) > 0:
                majority_label = Counter(points_in_cluster).most_common(1)[0][0]
                if majority_label == 'normal':
                    cluster_tags.append("Normal Traffic")
                else:
                    cluster_tags.append(f"Attack ({majority_label.upper()})")
            else:
                cluster_tags.append("Empty Cluster")
                
        return cluster_tags

    def get_optimal_centroids(self, k):
        """Uses scikit-learn to solve the absolute mathematical best positions instantly"""
        kmeans_model = KMeans(n_clusters=k, init='k-means++', n_init=10, max_iter=300, random_state=42)
        kmeans_model.fit(self.points)
        return kmeans_model.cluster_centers_

    def get_optimal_wcss(self, k):
        """Calculates scikit-learn's absolute best score benchmark for performance comparison"""
        kmeans_model = KMeans(n_clusters=k, init='k-means++', n_init=10, max_iter=300, random_state=42)
        kmeans_model.fit(self.points)
        return kmeans_model.inertia_

    def calculate_wcss(self):
        """Calculate the 'Game Score' (Inertia) with empty-cluster safety handles"""
        score = 0
        for i, c in enumerate(self.centroids):
            cluster_points = self.points[self.clusters == i]
            if len(cluster_points) > 0:
                score += np.sum((cluster_points - c)**2)
        return score

if __name__ == "__main__":
    engine = GSAILogic('data/KDDTrain+_20Percent.txt')
    print("Engine configured successfully.")