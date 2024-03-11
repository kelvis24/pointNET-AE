import argparse
import json
import torch
from PointCloudProcessor import PointCloudProcessor
from model import PointNetAutoencoder
from train import Train

def main():
    # Set up basic command-line argument parsing
    parser = argparse.ArgumentParser(description="Process and visualize point clouds.")
    parser.add_argument('-c', '--config', required=True, help='Path to the config JSON file')
    args = parser.parse_args()

    # Load the configuration file
    with open(args.config, 'r') as configFile:
        config = json.load(configFile)

    print('Config file loaded!\n')

    # Initialize device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')

    # Initialize the PointCloudProcessor
    processor = PointCloudProcessor(config["folder_dir"])

    # Optionally display loaded point clouds
    if config.get("show_loaded_pointclouds", "False").lower() == "true":
        for i, point_cloud in enumerate(processor.point_clouds):
            processor.plot_point_cloud(point_cloud, title=f"Point Cloud {i + 1}")

    # Initialize and train the model on all point clouds
    print("Training on all point clouds...")
    trainer = Train()

    model_trained_on_all_pointclouds = trainer.train_and_visualize_on_multiple_point_clouds_using_chamfer_distance_loss(
        processor.point_clouds, 
        num_points=config["min_points"], 
        latent_size=config["train_dim"], 
        epochs=config["epochs"], 
        lr=config["lr"], 
        visualize_every_n_epochs=config["visualize_every_n_epochs"], 
        condition=config.get("visualize_training", "False").lower() == "true"
    )

    # Test the model on a specific point cloud
    print("Testing model on a specific point cloud...")
    test_point_cloud = processor.load_point_cloud_from_ply(config["test_pointcloud_path"])
    test_point_cloud = processor.resample_point_cloud(test_point_cloud, config["min_points"])  # Resample to match min_points

    latent_representation = trainer.encode_point_cloud(model_trained_on_all_pointclouds, test_point_cloud.to(device))
    print("Latent vector shape of the test point cloud:", latent_representation.shape)
    print("Latent vector representation of the test point cloud:", latent_representation)

    # Example for plotting loss convergence for different latent sizes (if implemented in your Train class)
    latent_sizes = config["test_dims"]
    point_cloud_index = 0  # For demonstration; adjust as needed
    trainer.plot_loss_convergence_for_latent_sizes_using_log(
        latent_sizes, 
        config["min_points"], 
        processor.point_clouds, 
        point_cloud_index, 
        config["epochs"]
    )

if __name__ == "__main__":
    main()
