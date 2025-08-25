import cv2
import numpy as np
import torch
import os
import h5py
import einops
from torch.utils.data import TensorDataset, DataLoader

import IPython

e = IPython.embed


class EpisodicDataset(torch.utils.data.Dataset):
    def __init__(self, episode_ids, dataset_dir, camera_names, num_queries, norm_stats, load_images_to_memory,
                 ignore_depth, use_cameras, use_point_clouds, point_clouds, max_point_cloud_size):
        super(EpisodicDataset).__init__()
        self.episode_ids = episode_ids
        self.dataset_dir = dataset_dir
        # Name of the episode data file
        self.episode_file = [os.path.join(
            self.dataset_dir, f'episode_{episode_id}.hdf5') for episode_id in episode_ids]
        self.camera_names = camera_names
        self.use_cameras = use_cameras

        self.norm_stats = norm_stats
        self.is_sim = None
        self.num_queries = num_queries
        # Camera depth is a list of booleans indicating if the camera is a depth camera
        self.is_camera_depth = [cam_name.endswith(
            'RGBD') for cam_name in camera_names]
        # The number of cameras is the number of cameras plus the number of depth cameras
        if ignore_depth:
            self.num_cameras = len(camera_names)
        else:
            self.num_cameras = len(camera_names) + sum(self.is_camera_depth)
        self.use_point_clouds = use_point_clouds
        self.point_clouds = point_clouds
        self.max_point_cloud_size = max_point_cloud_size

        # Go through all episodes and load some data to memory
        self.is_sim_data = []
        self.qpos_data = []
        self.original_action_shape = []
        self.action_data = []
        self.image_data = []
        self.point_cloud_indices_data = {}

        self.load_images_to_memory = load_images_to_memory
        self.ignore_depth = ignore_depth

        for episode_id in self.episode_ids:
            dataset_path = os.path.join(
                self.dataset_dir, f'episode_{episode_id}.hdf5')
            with h5py.File(dataset_path, 'r') as root:
                is_sim = root.attrs['sim']
                self.is_sim_data.append(is_sim)

                qpos_name = '/observations/qpos'
                if qpos_name not in root:
                    qpos_name = '/observations/pos'
                if qpos_name not in root:
                    raise Exception(f'No qpos or pos in {dataset_path}')

                original_action_shape = root['/action'].shape
                self.original_action_shape.append(original_action_shape)

                # Get the qpos data
                qpos = root[qpos_name][()]
                qpos_data = torch.from_numpy(qpos).float()
                qpos_data = (
                    qpos_data - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]
                self.qpos_data.append(qpos_data)

                # Get the action data
                action_data = root['/action'][()]
                action_data = (
                    action_data - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]
                self.action_data.append(action_data)

                # Get the image data if asked to load to memory
                if self.load_images_to_memory:
                    self.image_data.append({cam_name: root[f'/observations/images/{cam_name}'][()] for cam_name in
                                            self.camera_names})

                # Get the point cloud indices data if we are using point clouds
                if self.use_point_clouds:
                    for point_cloud in self.point_clouds:
                        point_cloud_index = root[f'/observations/point_cloud/{point_cloud}_index'][(
                        )]
                        point_cloud_index_sum = point_cloud_index.cumsum()
                        self.point_cloud_indices_data[point_cloud] = point_cloud_index_sum

            print('Loaded episode', episode_id)

        # initialize self.is_sim
        self.__getitem__(0)

    def __len__(self):
        return len(self.episode_ids)

    def get_image_data_from_memory(self, index, start_ts):
        """
        Get the image data from memory
        """
        image_dict_episode = self.image_data[index]

        # Convert the image data into a dict for start_ts
        image_dict = dict()
        for cam_name in self.camera_names:
            image_dict[cam_name] = image_dict_episode[cam_name][start_ts]

        return image_dict

    def get_image_data_from_file(self, index, start_ts):
        """
        Get the image data from the file
        """
        image_dict = dict()
        dataset_path = self.episode_file[index]
        with h5py.File(dataset_path, 'r') as root:
            for cam_name in self.camera_names:
                image_dict[cam_name] = root[f'/observations/images/{cam_name}'][start_ts]

        return image_dict

    def get_image_data(self, index, start_ts):
        """
        Get the image data for the given episode index and timestep start_ts
        """

        # Get the image data from memory
        if self.load_images_to_memory:
            image_dict = self.get_image_data_from_memory(index, start_ts)
        else:
            image_dict = self.get_image_data_from_file(index, start_ts)

        # Preallocate list for all camera images from the number of cameras
        image_data = self.calculate_image_data(image_dict)

        # channel last
        image_data = torch.einsum('k h w c -> k c h w', image_data)

        # normalize image and change dtype to float
        image_data = image_data.float()
        image_data *= 1.0 / 255.0

        return image_data

    @staticmethod
    def get_point_cloud_padded_data(point_cloud_data, point_clouds, max_point_cloud_size):
        """
        Get the point cloud data and pad it to the max_point_cloud_size and 
        return it as numpy arrays of padded data and lengths
        Defining num_point_clouds as the number of cameras: 
        @return point_cloud_padded_data: (num_point_clouds, 3, max_point_cloud_size)
        @return point_cloud_data_len: (num_point_clouds)
        """

        # Data is in (N, 3) format. We have to pad it to make it (max_point_cloud_size, 3)
        point_cloud_padded_data = []
        point_cloud_data_len = []
        for point_cloud in point_clouds:
            point_cloud_data_len.append(len(point_cloud_data[point_cloud]))
            point_cloud_padded = np.pad(point_cloud_data[point_cloud], ((0, max_point_cloud_size - len(point_cloud_data[point_cloud])), (0, 0)),
                                        mode='constant', constant_values=0)
            point_cloud_padded_data.append(point_cloud_padded)

        # Reshape the padded data to be (num_point_clouds, 3, max_point_cloud_size) with the last dimension being the points
        point_cloud_padded_data = np.stack(point_cloud_padded_data, axis=0)
        point_cloud_padded_data = einops.rearrange(
            point_cloud_padded_data, 'num_point_clouds N C -> num_point_clouds C N')

        # Make the point cloud data len into a numpy array
        point_cloud_data_len = np.array(point_cloud_data_len)

        return point_cloud_padded_data, point_cloud_data_len

    def get_point_cloud_data(self, index, start_ts):
        """
        Get the point cloud data for the given episode index and timestep start_ts
        """
        # Get the point cloud data from the file using the indices which we have already loaded
        dataset_path = self.episode_file[index]
        point_cloud_data = dict()
        with h5py.File(dataset_path, 'r') as root:
            for point_cloud in self.point_clouds:
                index_data = self.point_cloud_indices_data[point_cloud]
                start_index = 0 if start_ts == 0 else index_data[start_ts - 1]
                end_index = index_data[start_ts]
                point_cloud_data[point_cloud] = root[
                    f'/observations/point_cloud/{point_cloud}'][start_index:end_index]

        point_cloud_padded_data, point_cloud_data_len = self.get_point_cloud_padded_data(
            point_cloud_data, self.point_clouds, self.max_point_cloud_size)

        return point_cloud_padded_data, point_cloud_data_len

    def calculate_image_data(self, image_dict):
        # Preallocate list for all camera images from the number of cameras
        image_0_size = image_dict[self.camera_names[0]].shape
        all_cam_images = np.empty(
            (self.num_cameras, image_0_size[0], image_0_size[1], 3), dtype=np.uint8)
        camera_idx = 0
        for cam_name in self.camera_names:
            cam_image = image_dict[cam_name]

            # Check if the last dimension is 4 meaning that it is a depth image
            if cam_image.shape[-1] == 4:
                # Take the first 3 channels and convert to color image
                rgb_image = cam_image[:, :, :3]
                all_cam_images[camera_idx] = rgb_image
                camera_idx += 1

                # Ignore depth images if the flag is set
                if self.ignore_depth:
                    continue

                depth_data = cam_image[:, :, 3]
                depth_image = cv2.applyColorMap(depth_data, cv2.COLORMAP_JET)
                all_cam_images[camera_idx] = depth_image
                camera_idx += 1
            else:
                all_cam_images[camera_idx] = cam_image
                camera_idx += 1
        # all_cam_images = np.stack(all_cam_images, axis=0)
        # construct observations
        image_data = torch.from_numpy(all_cam_images)
        return image_data

    def __getitem__(self, index):
        is_sim = self.is_sim_data[index]
        self.is_sim = is_sim

        original_action_shape = self.original_action_shape[index]
        episode_len = original_action_shape[0]

        start_ts = np.random.choice(episode_len)

        # get observation at start_ts only
        qpos = self.qpos_data[index][start_ts]

        # get all actions after and including start_ts
        if is_sim:
            action = self.action_data[index][start_ts:]
            action_len = episode_len - start_ts
        else:
            # hack, to make timesteps more aligned
            action = self.action_data[index][max(0, start_ts - 1):]
            action_len = episode_len - max(0, start_ts - 1)

        padded_action = np.ones(original_action_shape, dtype=np.float32) * (0 - self.norm_stats["action_mean"]) / \
            self.norm_stats["action_std"]
        padded_action[:action_len] = action
        padded_action = padded_action[:self.num_queries, :]
        action_data = torch.from_numpy(padded_action).float()

        # Create the is_pad tensor that has which values are real and which are padded
        is_pad = np.zeros(self.num_queries)
        is_pad[action_len:] = 1
        is_pad = torch.from_numpy(is_pad).bool()

        # Get the image data
        image_data = np.array([0])
        if self.use_cameras:
            image_data = self.get_image_data(index, start_ts)

        # Get the point cloud data
        point_cloud_data, point_cloud_data_len = np.array([0]), np.array([0])
        if self.use_point_clouds:
            point_cloud_data, point_cloud_data_len = self.get_point_cloud_data(
                index, start_ts)

        return image_data, point_cloud_data, point_cloud_data_len, qpos, action_data, is_pad

    def getitem_old(self, index):
        sample_full_episode = False  # hardcode

        episode_id = self.episode_ids[index]
        dataset_path = os.path.join(
            self.dataset_dir, f'episode_{episode_id}.hdf5')
        with h5py.File(dataset_path, 'r') as root:
            is_sim = root.attrs['sim']

            qpos_name = '/observations/qpos'
            if qpos_name not in root:
                qpos_name = '/observations/pos'
            if qpos_name not in root:
                raise Exception(f'No qpos or pos in {dataset_path}')

            original_action_shape = root['/action'].shape
            episode_len = original_action_shape[0]
            if sample_full_episode:
                start_ts = 0
            else:
                start_ts = np.random.choice(episode_len)
            # get observation at start_ts only
            qpos = root[qpos_name][start_ts]

            if 'qvel' in root['/observations']:
                qvel = root['/observations/qvel'][start_ts]
            image_dict = dict()
            for cam_name in self.camera_names:
                image_dict[cam_name] = root[f'/observations/images/{cam_name}'][start_ts]
            # get all actions after and including start_ts
            if is_sim:
                action = root['/action'][start_ts:]
                action_len = episode_len - start_ts
            else:
                # hack, to make timesteps more aligned
                action = root['/action'][max(0, start_ts - 1):]
                # hack, to make timesteps more aligned
                action_len = episode_len - max(0, start_ts - 1)

        self.is_sim = is_sim
        padded_action = np.zeros(original_action_shape, dtype=np.float32)
        padded_action[:action_len] = action
        is_pad = np.zeros(self.num_queries)
        is_pad[action_len:] = 1

        # new axis for different cameras
        all_cam_images = []
        for cam_name in self.camera_names:
            cam_image = image_dict[cam_name]

            # Check if the last dimension is 4 meaning that it is a depth image
            if cam_image.shape[-1] == 4:
                # Take the first 3 channels and convert to color image
                rgb_image = cam_image[:, :, :3]
                all_cam_images.append(rgb_image)
                depth_data = cam_image[:, :, 3]
                depth_image = cv2.applyColorMap(depth_data, cv2.COLORMAP_JET)
                all_cam_images.append(depth_image)
            else:
                all_cam_images.append(cam_image)

        all_cam_images = np.stack(all_cam_images, axis=0)

        # construct observations
        image_data = torch.from_numpy(all_cam_images)
        qpos_data = torch.from_numpy(qpos).float()
        action_data = torch.from_numpy(padded_action).float()
        is_pad = torch.from_numpy(is_pad).bool()

        # channel last
        image_data = torch.einsum('k h w c -> k c h w', image_data)

        # normalize image and change dtype to float
        image_data = image_data / 255.0
        action_data = (
            action_data - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]
        action_data = action_data[:self.num_queries, :]
        qpos_data = (
            qpos_data - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]

        return image_data, qpos_data, action_data, is_pad


def get_norm_stats(dataset_dir, num_episodes):
    all_qpos_data = []
    all_action_data = []
    for episode_idx in range(num_episodes):
        dataset_path = os.path.join(dataset_dir, f'episode_{episode_idx}.hdf5')
        with h5py.File(dataset_path, 'r') as root:
            qpos_name = '/observations/qpos'
            if qpos_name not in root:
                qpos_name = '/observations/pos'
            if qpos_name not in root:
                raise Exception(f'No qpos or pos in {dataset_path}')

            qpos = root[qpos_name][()]
            if 'qvel' in root['/observations']:
                qvel = root['/observations/qvel'][()]
            action = root['/action'][()]
        all_qpos_data.append(torch.from_numpy(qpos))
        all_action_data.append(torch.from_numpy(action))
    all_qpos_data = torch.vstack(all_qpos_data)
    all_action_data = torch.vstack(all_action_data)
    all_action_data = all_action_data

    # normalize action data
    action_mean = all_action_data.mean(dim=[0], keepdim=True)
    action_std = all_action_data.std(dim=[0], keepdim=True)
    action_std = torch.clip(action_std, 1e-2, np.inf)  # clipping

    # normalize qpos data
    qpos_mean = all_qpos_data.mean(dim=[0], keepdim=True)
    qpos_std = all_qpos_data.std(dim=[0], keepdim=True)
    qpos_std = torch.clip(qpos_std, 1e-2, np.inf)  # clipping

    stats = {"action_mean": action_mean.numpy().squeeze(), "action_std": action_std.numpy().squeeze(),
             "qpos_mean": qpos_mean.numpy().squeeze(), "qpos_std": qpos_std.numpy().squeeze(),
             "example_qpos": qpos}

    return stats


def load_data(dataset_dir, num_episodes, camera_names, num_queries, batch_size_train, batch_size_val,
              load_images_to_memory=False, ignore_depth=False, ignore_norm=[], use_images=True, use_point_clouds=False, point_clouds=[], max_point_cloud_size=0):
    print(f'\nData from: {dataset_dir} with {num_episodes} episodes\n')
    # obtain train test split
    train_ratio = 0.8
    shuffled_indices = np.random.permutation(num_episodes)
    train_indices = shuffled_indices[:int(train_ratio * num_episodes)]
    val_indices = shuffled_indices[int(train_ratio * num_episodes):]

    # obtain normalization stats for qpos and action
    norm_stats = get_norm_stats(dataset_dir, num_episodes)

    # Set the normalization stats to mean 0 and std 1 for the ignored values so original values remain
    for ignore_key in ignore_norm:
        norm_stats['action_mean'][ignore_key] = 0
        norm_stats['action_std'][ignore_key] = 1
        norm_stats['qpos_mean'][ignore_key] = 0
        norm_stats['qpos_std'][ignore_key] = 1

    # construct dataset and dataloader
    train_dataset = EpisodicDataset(train_indices, dataset_dir, camera_names, num_queries, norm_stats,
                                    load_images_to_memory, ignore_depth, use_images, use_point_clouds, point_clouds, max_point_cloud_size)
    val_dataset = EpisodicDataset(val_indices, dataset_dir, camera_names, num_queries, norm_stats,
                                  load_images_to_memory, ignore_depth, use_images, use_point_clouds, point_clouds, max_point_cloud_size)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, pin_memory=True,
                                  num_workers=0, prefetch_factor=None)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=True, pin_memory=True, num_workers=0,
                                prefetch_factor=None)

    return train_dataloader, val_dataloader, norm_stats, train_dataset.is_sim


# env utils

def sample_box_pose():
    x_range = [0.0, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    cube_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    cube_quat = np.array([1, 0, 0, 0])
    return np.concatenate([cube_position, cube_quat])


def sample_insertion_pose():
    # Peg
    x_range = [0.1, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    peg_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    peg_quat = np.array([1, 0, 0, 0])
    peg_pose = np.concatenate([peg_position, peg_quat])

    # Socket
    x_range = [-0.2, -0.1]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    socket_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    socket_quat = np.array([1, 0, 0, 0])
    socket_pose = np.concatenate([socket_position, socket_quat])

    return peg_pose, socket_pose


# helper functions

def compute_dict_mean(epoch_dicts):
    result = {k: None for k in epoch_dicts[0]}
    num_items = len(epoch_dicts)
    for k in result:
        value_sum = 0
        for epoch_dict in epoch_dicts:
            value_sum += epoch_dict[k]
        result[k] = value_sum / num_items
    return result


def detach_dict(d):
    new_d = dict()
    for k, v in d.items():
        new_d[k] = v.detach()
    return new_d


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def depth_to_8bit(depth_data):
    """
    Convert depth image to 8 bit image for storage
    """

    # Normalize depth data so the closest point is 0
    depth_data -= depth_data.min()
    # Scale by 2 mean distances of near rays
    depth_data /= 2 * depth_data[depth_data <= 1].mean()
    # Clip to 0-1
    depth_data = np.clip(depth_data, 0, 1)
    # Convert to uint8 with range 0-255
    depth_data_uint8 = (depth_data * 255).astype(np.uint8)

    return depth_data_uint8


def depth_to_rgb(depth_data):
    """
    Convert depth image to RGB image. The depth is given as a float32 of depth in meters. The output is a uint8 image
    where the depth is mapped to the jet colormap.
    """

    # Convert to uint8 with range 0-255
    depth_data_uint8 = depth_to_8bit(depth_data)
    # Apply colormap JET
    color_image = cv2.applyColorMap(depth_data_uint8, cv2.COLORMAP_JET)

    return color_image


def load_from_args(args, arg_name, default):
    """
    Load a value from the args dictionary. If the value is not present, return the default value.
    """
    if arg_name in args:
        return args[arg_name]
    else:
        return default
