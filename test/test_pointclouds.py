import h5py
import torch

from pointclouds.pointnet import PointNet


def test_point_net():
    # Set the name for the dataset
    dataset_dir = "D:\\Share\\PC-Only"
    episode_idx = 0
    episode_hdf_file = f"{dataset_dir}/episode_{episode_idx}.hdf5"

    model = PointNet()
    model = model.cuda()

    with h5py.File(episode_hdf_file, "r") as f:
        # Get the number of time steps
        num_time_steps = 1

        # Get the name of the cameras
        point_cloud_keys = list(f["observations/point_cloud"].keys())

        # Separate the point cloud data keys into ones that end with index and one without
        point_cloud_index = [
            key for key in point_cloud_keys if key.endswith("_index")]
        came_names = [
            key for key in point_cloud_keys if key not in point_cloud_index]

        # For each camera, set up point cloud indices for each time step
        point_cloud_indices = dict()
        for cam_name in came_names:
            index_data = f["observations/point_cloud"][f"{cam_name}_index"][:]
            cummulative_index_data = index_data.cumsum()
            point_cloud_indices[cam_name] = cummulative_index_data

        # Get the point cloud data for the given time step
        for time_step in range(num_time_steps):
            point_cloud_data = dict()
            for cam_name in came_names:
                start_index = (
                    0
                    if time_step == 0
                    else point_cloud_indices[cam_name][time_step - 1]
                )
                end_index = point_cloud_indices[cam_name][time_step]
                point_cloud_data[cam_name] = f["observations/point_cloud"][cam_name][
                    start_index:end_index
                ]

            # Get the embeddings for the point cloud data using point net
            for cam_name, point_cloud in point_cloud_data.items():
                point_cloud = torch.tensor(
                    point_cloud).transpose(1, 0).unsqueeze(0)
                point_cloud = point_cloud.cuda()
                # _, _, _, feature_map = model(point_cloud)
                feature_map = model(point_cloud)
                assert feature_map['feature_map'].shape == (1, 512, 1, 1)
                print(f"Tested for {cam_name} at time step {time_step}")
