import pandas as pd
import numpy as np

class GSAILogic:
    def __init__(self, file_path):
        # Load and prepare data
        self.df = pd.read_csv(file_path, header=None)
        
        # Better scaling for NSL-KDD traffic data
        # We take the log of the bytes to handle the huge variance in network traffic
        self.points = np.log1p(self.df.iloc[:500, [4, 5]].values) 
        
        # Now normalize that logged data to 0.1 - 0.9 so it's not touching the edges
        self.points = 0.1 + 0.8 * (self.points - self.points.min(axis=0)) / (self.points.max(axis=0) - self.points.min(axis=0))
        
        self.centroids = []
        self.clusters = []

    def initialize_centroids(self, k):
        """Randomly pick K points to start (Standard K-Means)"""
        indices = np.random.choice(len(self.points), k, replace=False)
        self.centroids = self.points[indices]
        return self.centroids

    def step_calculate_clusters(self):
        """Phase 1: Assign each point to the nearest centroid"""
        distances = np.linalg.norm(self.points[:, np.newaxis] - self.centroids, axis=2)
        self.clusters = np.argmin(distances, axis=1)
        return self.clusters

    def step_move_centroids(self):
        """Phase 2: Move centroids to the mean of their clusters"""
        new_centroids = np.array([self.points[self.clusters == i].mean(axis=0) 
                                 if len(self.points[self.clusters == i]) > 0 else self.centroids[i]
                                 for i in range(len(self.centroids))])
        self.centroids = new_centroids
        return self.centroids

    def calculate_wcss(self):
        """Calculate the 'Game Score' (Inertia)"""
        score = 0
        for i, c in enumerate(self.centroids):
            cluster_points = self.points[self.clusters == i]
            score += np.sum((cluster_points - c)**2)
        return score

# Quick test
if __name__ == "__main__":
    engine = GSAILogic('data/KDDTrain+_20Percent.txt')
    print("Engine ready with", len(engine.points), "data points.")