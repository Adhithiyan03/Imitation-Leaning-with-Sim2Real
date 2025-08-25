import torch
import numpy as np
from torch.utils.data import DataLoader

from utils import EpisodicDataset, get_norm_stats


def test_EpisodicDataset():
    """
    Create the EpisodicDataset object and then get a sample from it. Test if the sample is of the correct shape.
    """
    # Create the dataset
    camera_names = ['cam_left', 'cam_right',
                    'cam_top_RGBD', 'cam_wrist', 'cam_wrist_RGBD']
    chunk_size = 100
    dataset_dir = 'D:\\PiH_Depth'
    num_episodes = 2
    norm_stats = get_norm_stats(dataset_dir, num_episodes)
    dataset = EpisodicDataset(episode_ids=[0, 1], dataset_dir='D:\\PiH_Depth', camera_names=camera_names,
                              num_queries=chunk_size, norm_stats=norm_stats, load_images_to_memory=False, ignore_depth=False,
                              use_cameras=True, use_point_clouds=False, point_clouds=[], max_point_cloud_size=0)

    # Get a sample from the dataset
    sample = dataset[0]

    # Check if the sample is of the correct shape
    assert sample[0].shape == (7, 3, 480, 640)
    assert sample[1].shape == (1, )
    assert sample[2].shape == (1, )
    assert sample[3].shape == (7,)
    assert sample[4].shape == (100, 7)
    assert sample[5].shape == (100,)

    num_samples = 10
    for i in range(num_samples):
        sample = dataset[0]
    for i in range(num_samples):
        sample = dataset[1]


def test_EpisodicDataset_point_clouds():
    """
    Test that the point clouds are being loaded correctly in the dataset. 
    This is for hdf5 files that do not have any camera images.
    """
    dataset_dir = "D:\\Share\\PC-Only"
    camera_names = []
    chunk_size = 100
    num_episodes = 2
    norm_stats = get_norm_stats(dataset_dir, num_episodes)
    max_point_cloud_size = 640 * 480
    point_clouds = ['top', 'angle']

    # Create the dataset
    dataset = EpisodicDataset(episode_ids=[0, 1], dataset_dir=dataset_dir, camera_names=camera_names,
                              num_queries=chunk_size, norm_stats=norm_stats, load_images_to_memory=False, ignore_depth=False,
                              use_cameras=False, use_point_clouds=True, point_clouds=point_clouds, max_point_cloud_size=max_point_cloud_size)

    # Check that the cloud index data has been loaded
    assert dataset.point_cloud_indices_data is not None

    # Check that point cloud data is a dict and has the keys from the point_clouds list
    assert isinstance(dataset.point_cloud_indices_data, dict)
    assert set(dataset.point_cloud_indices_data.keys()) == set(point_clouds)

    # Get a sample from the dataset
    image_data, point_cloud_data, point_cloud_data_len, qpos, action_data, is_pad = dataset[1]

    # Since the image_data is not loaded, it should be a single value of 0
    assert image_data == 0
    assert point_cloud_data.shape == (
        len(point_clouds), 3, max_point_cloud_size)
    assert point_cloud_data_len.shape == (len(point_clouds),)
    assert qpos.shape == (8,)
    assert action_data.shape == (chunk_size, 8)
    assert is_pad.shape == (chunk_size,)


def test_EpisodicDataset_point_clouds_dataloader():
    """
    Load a dataset with only point clouds and test if the dataloader is working correctly.
    """
    dataset_dir = "D:\\Share\\PC-Only"
    camera_names = []
    chunk_size = 100
    num_episodes = 2
    norm_stats = get_norm_stats(dataset_dir, num_episodes)
    max_point_cloud_size = 640 * 480
    point_clouds = ['top', 'angle']

    # Create the dataset
    dataset = EpisodicDataset(episode_ids=[0, 1], dataset_dir=dataset_dir, camera_names=camera_names,
                              num_queries=chunk_size, norm_stats=norm_stats, load_images_to_memory=False, ignore_depth=False,
                              use_cameras=False, use_point_clouds=True, point_clouds=point_clouds, max_point_cloud_size=max_point_cloud_size)

    # Create the dataloader
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    # Load the data from the dataloader
    for i, data in enumerate(dataloader):
        image_data, point_cloud_data, point_cloud_data_len, qpos, action_data, is_pad = data

        # Check that image data is a tensor of zeros of size (2, 1)
        assert torch.all(image_data == 0)
        assert image_data.shape == (2, 1)

        # Check the point cloud data is the right shape
        assert point_cloud_data.shape == (
            2, len(point_clouds), 3, max_point_cloud_size)
        assert point_cloud_data_len.shape == (2, len(point_clouds))

        # Check that the qpos and action data are the right shape
        assert qpos.shape == (2, 8)
        assert action_data.shape == (2, chunk_size, 8)
        assert is_pad.shape == (2, chunk_size)


def test_stack_removal():
    """
    Create the Episodic dataset and test if the stack removal is working as expected.
    """
    camera_names = ['cam_left', 'cam_right',
                    'cam_top_RGBD', 'cam_wrist', 'cam_wrist_RGBD']
    chunk_size = 100
    dataset_dir = 'D:\\PiH_Depth'
    num_episodes = 2
    norm_stats = get_norm_stats(dataset_dir, num_episodes)
    dataset = EpisodicDataset(episode_ids=[0, 1], dataset_dir='D:\\PiH_Depth', camera_names=camera_names,
                              num_queries=chunk_size, norm_stats=norm_stats)

    for random_seed in range(10):
        print('Random seed: ', random_seed)

        # Set the seed for reproducibility in numpy
        np.random.seed(random_seed)

        # Get a sample from the dataset
        sample = dataset[0]

        # Get the sample from the old method
        np.random.seed(random_seed)
        sample_old = dataset.getitem_old(0)

        # Check that old sample and new sample are the same
        assert np.allclose(sample[0], sample_old[0])
        assert np.allclose(sample[1], sample_old[1])
        assert np.allclose(sample[2], sample_old[2])
        assert np.allclose(sample[3], sample_old[3])
