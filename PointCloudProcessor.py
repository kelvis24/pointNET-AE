# Import necessary libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from plyfile import PlyData
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from torch.optim import Adam
from sklearn.decomposition import PCA
from torch.optim import Adam
device="cuda"

class PointCloudProcessor:
    def __init__(self, point_clouds_directory):
        self.point_clouds_directory = point_clouds_directory
        self.point_clouds = self.load_and_preprocess_point_clouds()

    def load_point_cloud_from_ply(self, file_path):
        ply_data = PlyData.read(file_path)
        points = np.column_stack((ply_data['vertex']['x'], ply_data['vertex']['y'], ply_data['vertex']['z']))
        return torch.tensor(points, dtype=torch.float)

    def resample_point_cloud(self, point_cloud, num_points):
        if len(point_cloud) == num_points:
            return point_cloud
        elif len(point_cloud) > num_points:
            indices = np.random.choice(len(point_cloud), size=num_points, replace=False)
            return point_cloud[indices]
        else:
            raise ValueError("Point cloud has fewer points than required for resampling.")

    def plot_point_cloud(self, points, title="Point Cloud", ax=None):
        if ax is None:
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111, projection='3d')
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1)  # 's' is the size of each point
        ax.set_title(title)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.show()

    def visualize_reconstruction(self, original, reconstructed, title1="Original", title2="Reconstructed"):
        fig = plt.figure(figsize=(12, 6))

        ax1 = fig.add_subplot(121, projection='3d')
        ax1.scatter(original[:, 0], original[:, 1], original[:, 2], s=1)
        ax1.title.set_text(title1)
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')

        ax2 = fig.add_subplot(122, projection='3d')
        ax2.scatter(reconstructed[:, 0], reconstructed[:, 1], reconstructed[:, 2], s=1)
        ax2.title.set_text(title2)
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')

        plt.show()

    def load_and_preprocess_point_clouds(self):
        point_cloud_files = [f for f in os.listdir(self.point_clouds_directory) if f.endswith('.ply')]
        point_clouds = [self.load_point_cloud_from_ply(os.path.join(self.point_clouds_directory, f)) for f in point_cloud_files]

        # Resample all point clouds to have the same number of points
        self.min_points = min(pc.shape[0] for pc in point_clouds)
        return [self.resample_point_cloud(pc, self.min_points) for pc in point_clouds]



if __name__ == "__main__":
    # Example usage:
    point_clouds_directory = './simple-3d-objects'
    processor = PointCloudProcessor(point_clouds_directory)

    for i, point_cloud in enumerate(processor.point_clouds):
        processor.plot_point_cloud(point_cloud, title=f"Point Cloud {i + 1}")