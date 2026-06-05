import pandas as pd
import numpy as np

class GSAILogic:
    def __init__(self, file_path):
        # Load and prepare data
        self.df = pd.read_csv(file_path, header=None)
        
        # Better scaling for NSL-KDD traffic data
        self.points = np.log1p(self.df.iloc[:500, [4, 5]].values) 
        
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
        
        # Check if positions completely stopped changing
        has_converged = np.allclose(old_centroids, new_centroids, atol=1e-6)
        return has_converged

    def calculate_wcss(self):
        """Calculate the 'Game Score' (Inertia) with empty-cluster safety handles"""
        score = 0
        for i, c in enumerate(self.centroids):
            cluster_points = self.points[self.clusters == i]
            if len(cluster_points) > 0:
                score += np.sum((cluster_points - c)**2)
        return score

# Quick test
if __name__ == "__main__":
    engine = GSAILogic('data/KDDTrain+_20Percent.txt')
    print("Engine ready with", len(engine.points), "data points.")