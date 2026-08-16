import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

class GSAILogic:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.raw_df = None
        self.points = []
        self.point_labels = []
        self.centroids = []
        self.clusters = []
        self.current_level = 1
        
        # Load raw dataset
        self.load_raw_data()
        
        # Initialize default level data (Level 1)
        self.load_level(1)

    def load_raw_data(self):
        """Loads NSL-KDD dataset with proper column headers."""
        columns = [
            'duration', 'protocol_type', 'service', 'flag', 'src_bytes',
            'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot',
            'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell',
            'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
            'num_access_files', 'num_outbound_cmds', 'is_host_login',
            'is_guest_login', 'count', 'srv_count', 'serror_rate',
            'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate',
            'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate',
            'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate',
            'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
            'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
            'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
            'dst_host_srv_rerror_rate', 'attack_class', 'difficulty'
        ]
        
        try:
            self.raw_df = pd.read_csv(self.dataset_path, names=columns, header=None)
        except Exception as e:
            print(f"Error loading dataset: {e}")
            self.raw_df = pd.DataFrame()

    def load_level(self, level_num):
        """Filters dataset for specific attack categories based on level selected."""
        self.current_level = level_num
        
        level_profiles = {
            1: ['normal', 'neptune'],
            2: ['normal', 'neptune', 'back'],
            3: ['normal', 'portsweep', 'neptune'],
            4: ['normal', 'neptune', 'smurf', 'satan'],
            5: ['normal', 'neptune', 'satan', 'warezclient', 'teardrop']
        }
        
        classes = level_profiles.get(level_num, level_profiles[1])
        
        # Filter dataframe for target attack classes
        filtered_df = self.raw_df[self.raw_df['attack_class'].isin(classes)].copy()
        
        # Sample points per class to balance visualization on canvas
        sampled_frames = []
        for c in classes:
            c_df = filtered_df[filtered_df['attack_class'] == c]
            sample_size = min(len(c_df), 120 if c == 'normal' else 70)
            if sample_size > 0:
                sampled_frames.append(c_df.sample(n=sample_size, random_state=42 + level_num))
                
        if sampled_frames:
            df_subset = pd.concat(sampled_frames).sample(frac=1, random_state=42).reset_index(drop=True)
        else:
            df_subset = filtered_df.sample(n=min(len(filtered_df), 300), random_state=42).reset_index(drop=True)

        # Log transform continuous network traffic features for 2D spatial mapping
        x_raw = np.log1p(df_subset['src_bytes'].values.astype(float))
        y_raw = np.log1p(df_subset['count'].values.astype(float))

        # Min-Max Normalization to fit [0.05, 0.95] canvas boundaries
        x_min, x_max = x_raw.min(), x_raw.max()
        y_min, y_max = y_raw.min(), y_raw.max()
        
        norm_x = 0.05 + 0.90 * ((x_raw - x_min) / (x_max - x_min + 1e-6))
        norm_y = 0.05 + 0.90 * ((y_raw - y_min) / (y_max - y_min + 1e-6))

        # Store normalized (X, Y) points and corresponding labels
        self.points = np.column_stack((norm_x, norm_y))
        self.point_labels = df_subset['attack_class'].tolist()
        
        # Reset cluster tracking arrays
        self.centroids = []
        self.clusters = []

    def step_calculate_clusters(self):
        """Assigns each point to the nearest centroid."""
        if len(self.centroids) == 0:
            return
            
        distances = np.linalg.norm(self.points[:, np.newaxis] - np.array(self.centroids), axis=2)
        self.clusters = np.argmin(distances, axis=1)

    def step_move_centroids(self):
        """Recalculates centroid coordinates based on mean of assigned points."""
        if len(self.clusters) == 0:
            return False
            
        new_centroids = []
        for i in range(len(self.centroids)):
            assigned_points = self.points[self.clusters == i]
            if len(assigned_points) > 0:
                new_centroids.append(assigned_points.mean(axis=0))
            else:
                new_centroids.append(self.centroids[i])
                
        new_centroids = np.array(new_centroids)
        
        # Check convergence (movement threshold < 1e-4)
        diff = np.linalg.norm(self.centroids - new_centroids)
        self.centroids = new_centroids
        return diff < 1e-4

    def calculate_wcss(self):
        """Calculates Within-Cluster Sum of Squares score."""
        if len(self.clusters) == 0 or len(self.centroids) == 0:
            return 0.0
        
        wcss = 0.0
        for i, p in enumerate(self.points):
            c_idx = self.clusters[i]
            centroid = self.centroids[c_idx]
            wcss += np.sum((p - centroid) ** 2)
        return float(wcss)

    def get_cluster_labels_and_colors(self):
        """Maps majority attack class and color scheme to active clusters."""
        if len(self.clusters) == 0 or len(self.centroids) == 0:
            return [], []
            
        labels = []
        colors = []
        
        palette = ["#FF3366", "#33CCFF", "#FFCC00", "#9966FF", "#FF9933"]
        
        for i in range(len(self.centroids)):
            assigned_indices = np.where(self.clusters == i)[0]
            if len(assigned_indices) > 0:
                cluster_classes = [self.point_labels[idx] for idx in assigned_indices]
                majority_class = max(set(cluster_classes), key=cluster_classes.count)
                
                if majority_class == "normal":
                    lbl = "Normal Traffic"
                    col = "#39FF14"  # Bright Green
                else:
                    lbl = f"Attack ({majority_class.upper()})"
                    col = palette[i % len(palette)]
            else:
                lbl = "Empty Cluster"
                col = "#888888"
                
            labels.append(lbl)
            colors.append(col)
            
        return labels, colors

    def get_cluster_labels(self):
        labels, _ = self.get_cluster_labels_and_colors()
        return labels

    def get_optimal_centroids(self, k):
        """Computes ground truth optimal centroids using scikit-learn KMeans."""
        if len(self.points) == 0:
            return []
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(self.points)
        return kmeans.cluster_centers_

    def get_optimal_wcss(self, k):
        """Returns optimal WCSS inertia for scoring comparisons."""
        if len(self.points) == 0:
            return 0.0
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(self.points)
        return float(kmeans.inertia_)