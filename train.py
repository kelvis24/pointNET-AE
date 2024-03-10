import torch
import torch.nn as nn
from torch.optim import Adam
import matplotlib.pyplot as plt
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
from PointCloudProcessor import PointCloudProcessor

from model import PointNetAutoencoder

class Train:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def train_and_visualize_on_single_point_cloud(self, point_cloud, epochs=100, lr=0.001, visualize_every_n_epochs=20, condition=False):
            autoencoder = PointNetAutoencoder().to(self.device)
            optimizer = Adam(autoencoder.parameters(), lr=lr)
            criterion = nn.MSELoss()

            point_cloud_tensor = point_cloud.unsqueeze(0).to( self.device)  # Ensure it's in the correct shape for training

            for epoch in range(epochs):
                # Forward pass
                reconstructed, _ = autoencoder(point_cloud_tensor)
                loss = criterion(reconstructed, point_cloud_tensor)

                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if (epoch + 1) % visualize_every_n_epochs == 0 or epoch == epochs - 1:
                    print(f"Epoch {epoch + 1}, Loss: {loss.item()}")
                    with torch.no_grad():
                        reconstructed, _ = autoencoder(point_cloud_tensor)
                        self.visualize_reconstruction(point_cloud_tensor.squeeze().cpu().numpy(),
                                                reconstructed.squeeze().cpu().numpy(),
                                                "Original", "Reconstructed", condition)

            return autoencoder


    def train_and_visualize_on_multiple_point_clouds(self, point_clouds, num_points=500, latent_size=218, epochs=100, lr=0.001, visualize_every_n_epochs=20, condition=False):
            criterion = nn.MSELoss()

            for pc_index, point_cloud in enumerate(point_clouds):
                autoencoder = PointNetAutoencoder(num_points, latent_size).to( self.device)
                optimizer = Adam(autoencoder.parameters(), lr=lr)
                point_cloud_tensor = point_cloud.unsqueeze(0).to( self.device)  # Add batch dimension

                print(f"Training Point Cloud {pc_index + 1}/{len(point_clouds)}")

                for epoch in range(epochs):
                    # Forward pass
                    reconstructed, _ = autoencoder(point_cloud_tensor)
                    loss = criterion(reconstructed, point_cloud_tensor)

                    # Backward pass and optimize
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    if (epoch + 1) % visualize_every_n_epochs == 0 or epoch == epochs - 1:
                        print(f"Epoch {epoch + 1}, Loss: {loss.item()}")
                        with torch.no_grad():
                            reconstructed, _ = autoencoder(point_cloud_tensor)
                            self.visualize_reconstruction(point_cloud_tensor.squeeze().cpu().numpy(),
                                                    reconstructed.squeeze().cpu().numpy(),
                                                    "Original", "Reconstructed", condition)

            return autoencoder


    def train_and_visualize_using_different_autoencoders_on_multiple_pointclouds(self, autoencoder, point_clouds, epochs=100, lr=0.001, visualize_every_n_epochs=20, condition=False):
            criterion = nn.MSELoss()
            optimizer = Adam(autoencoder.parameters(), lr=lr)

            for pc_index, point_cloud in enumerate(point_clouds):
                point_cloud_tensor = point_cloud.unsqueeze(0).to( self.device)  # Add batch dimension

                print(f"Training Point Cloud {pc_index + 1}/{len(point_clouds)} with Autoencoder")

                for epoch in range(epochs):
                    # Forward pass
                    reconstructed, _ = autoencoder(point_cloud_tensor)
                    loss = criterion(reconstructed, point_cloud_tensor)

                    # Backward pass and optimize
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    if (epoch + 1) % visualize_every_n_epochs == 0 or epoch == epochs - 1:
                        print(f"Epoch {epoch + 1}, Loss: {loss.item()}")
                        with torch.no_grad():
                            reconstructed, _ = autoencoder(point_cloud_tensor)
                            self.visualize_reconstruction(point_cloud_tensor.squeeze().cpu().numpy(),
                                                    reconstructed.squeeze().cpu().numpy(),
                                                    "Original", "Reconstructed", condition)

            return autoencoder


    def train_and_visualize_using_different_autoencoders_on_single_pointcloud(self, autoencoder, point_cloud, epochs=100, lr=0.001, visualize_every_n_epochs=20, condition=False):
            autoencoder.to( self.device)
            criterion = nn.MSELoss()
            optimizer = Adam(autoencoder.parameters(), lr=lr)

            # Ensure point_cloud is on the correct device
            point_cloud = point_cloud.to( self.device).unsqueeze(0)  # Ensure there is a batch dimension

            for epoch in range(epochs):
                # Forward pass
                reconstructed, _ = autoencoder(point_cloud)
                loss = criterion(reconstructed, point_cloud)

                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Visualization and logging
                if (epoch + 1) % visualize_every_n_epochs == 0 or (epoch + 1) == epochs:
                    print(f"Epoch {epoch+1}, Loss: {loss.item()}")
                    with torch.no_grad():
                        original = point_cloud.squeeze().cpu().numpy()  # Use the point cloud for visualization
                        reconstructed, _ = autoencoder(point_cloud)
                        reconstructed = reconstructed.squeeze().cpu().numpy()
                        self.visualize_reconstruction(original, reconstructed, "Original", "Reconstructed", condition)
            return autoencoder


    def train_and_return_losses(self, autoencoder, point_cloud, epochs=100, lr=0.001):
        optimizer = Adam(autoencoder.parameters(), lr=lr)
        criterion = nn.MSELoss()
        losses = []

        point_cloud = point_cloud.to(self.device).unsqueeze(0)

        for epoch in range(epochs):
            autoencoder.train()
            optimizer.zero_grad()
            reconstructed, _ = autoencoder(point_cloud)
            loss = criterion(reconstructed, point_cloud)
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        return losses

    def plot_loss_convergence_for_latent_sizes(self, latent_sizes, min_points, point_clouds, point_cloud_index, epochs=100, lr=0.001):
        loss_records = {}

        for latent_size in latent_sizes:
            print(f"Training with latent size: {latent_size}")
            # Assuming the autoencoder model can be re-initialized with different latent sizes
            autoencoder = PointNetAutoencoder(num_points=min_points, latent_size=latent_size).to(self.device)

            losses = self.train_and_return_losses(autoencoder, point_clouds[point_cloud_index] , epochs, lr)
            loss_records[latent_size] = losses

        plt.figure(figsize=(10, 7))
        for latent_size, losses in loss_records.items():
            plt.plot(losses, label=f'Latent Size {latent_size}')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss Convergence for Different Latent Sizes')
        plt.legend()
        plt.show()

    def encode_point_cloud(self, autoencoder, point_cloud):
         # Reshape selected_point_cloud to [1, 3, num_points] for the model
        if point_cloud.dim() == 2 and point_cloud.size(1) == 3:
            model_input = point_cloud.transpose(0, 1).unsqueeze(0)
        else:
            print("Unexpected point cloud shape. Ensure it's [num_points, 3].")

        with torch.no_grad():
            autoencoder.eval()
            point_cloud = point_cloud.to(self.device).unsqueeze(0)
            _, latent_representation = autoencoder(point_cloud)
            return latent_representation.squeeze().cpu().numpy()

    def print_encode_point_cloud(self, autoencoder, point_cloud):
         print(self.encode_point_cloud(autoencoder, point_cloud))

    def visualize_reconstruction(self, original, reconstructed, title1="Original", title2="Reconstructed", condition=True):
        if condition:
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
        else:
          pass


def main():
    # Initialize the training class
    trainer = Train()

    point_clouds_directory = '/Users/elviskimara/Downloads/PointNET Baseline/data'
    processor = PointCloudProcessor(point_clouds_directory)

    # Example of training and visualizing on a single point cloud
    trainer.train_and_visualize_on_single_point_cloud(processor.point_clouds[9], epochs=50, lr=0.001, visualize_every_n_epochs=10, condition=True)

if __name__ == "__main__":
    main()