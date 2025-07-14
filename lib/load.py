import pandas as pd
import matplotlib.tri as tri
import numpy as np
import itertools
import tqdm

import torch
from torch_geometric.data import Data
from torch_geometric.transforms import AddLaplacianEigenvectorPE


class GraphDataset:
    def __init__(
        self,
        training_clusters,
        dataset_size=512,
        noise_point_range=(150, 250),
        cluster_count_range=(10, 20),
        cluster_sampling_range=(0.055, 0.062),
        connectivity_radius=1.0,
        pupulation_std_sigma=0.003,
        num_additions=4,
    ):
        self.training_clusters = training_clusters
        self.dataset_size = dataset_size
        self.noise_point_range = noise_point_range
        self.cluster_count_range = cluster_count_range
        self.cluster_sampling_range = cluster_sampling_range
        self.connectivity_radius = connectivity_radius
        self.pupulation_std_sigma = pupulation_std_sigma

        self.laplacian_embedding = AddLaplacianEigenvectorPE(
            5, attr_name="x", is_undirected=True
        )

        self.training_data = [
            self.populate_cluster(cluster, self.pupulation_std_sigma, num_additions)
            for cluster in tqdm.tqdm_notebook(training_clusters)
        ]

    def __call__(self):
        return self.generate_dataset()

    def generate_dataset(self):
        """Generate a dataset of graph-structured data."""
        dataset = []

        for _ in tqdm.tqdm_notebook(
            range(self.dataset_size), desc="Generating dataset"
        ):
            clusters, cluster_shifts = self.generate_clusters()
            noise_points = self.generate_noise_points()

            positions = np.concatenate([clusters, noise_points], axis=0)
            deltas = np.concatenate(cluster_shifts, axis=0)

            data = self.create_data_object(positions, deltas)
            dataset.append(data)

        return dataset

    def generate_clusters(self):
        """Generate clustered data points with random shifts and rotations."""
        num_clusters = np.random.randint(*self.cluster_count_range)
        random_shifts = np.random.uniform(0.05, 0.95, size=(num_clusters, 2))

        all_positions = []
        all_deltas = []

        for shift in random_shifts:
            cluster = self.sample_random_cluster()
            rotated_cluster = self.apply_random_rotation(cluster)
            shifted_positions = rotated_cluster + shift

            all_positions.append(shifted_positions)
            all_deltas.append(rotated_cluster)

        return np.concatenate(all_positions, axis=0), all_deltas

    def populate_cluster(self, cluster, sigma, num_additions):
        cluster = cluster.copy()
        for i in range(num_additions):
            addition = np.random.normal(0, sigma, size=(cluster.shape[0], 2))
            addition = cluster[["x", "y"]].values + addition
            cluster = pd.concat([cluster, pd.DataFrame(addition, columns=["x", "y"])])

        return cluster

    def sample_random_cluster(self):
        """Sample a subset of the training data to represent a cluster."""
        cluster = self.training_data[
            np.random.randint(0, len(self.training_data))
        ].copy()
        return cluster.sample(frac=np.random.uniform(*self.cluster_sampling_range))[
            ["x", "y"]
        ].values

    @staticmethod
    def apply_random_rotation(points):
        """Apply a random rotation to a set of 2D points."""
        angle = np.random.uniform(0, 2 * np.pi)
        rotation_matrix = np.array(
            [
                [np.cos(angle), -np.sin(angle)],
                [np.sin(angle), np.cos(angle)],
            ]
        )
        return np.dot(points, rotation_matrix)

    def generate_noise_points(self):
        """Generate random noise points."""
        num_noise_points = np.random.randint(*self.noise_point_range)
        return np.random.uniform(0, 1, size=(num_noise_points, 2))

    def create_data_object(self, positions, deltas):
        """Create a `Data` object from positions and deltas."""
        data = Data()

        # Prepare labels
        noise_labels = np.zeros_like(positions[len(deltas) :])  # For noise
        data.y = (
            torch.tensor(
                np.concatenate([deltas, noise_labels], axis=0), dtype=torch.float
            )
            / self.connectivity_radius
        )

        # Add graph structure
        data.edge_index, data.edge_attr = self.compute_connectivity(positions)
        data.num_nodes = positions.shape[0]

        # Add positional and embedding features
        data = self.laplacian_embedding(data)
        data.x = torch.abs(data.x)
        data.position = torch.tensor(positions, dtype=torch.float)

        return data

    def compute_connectivity(self, positions):
        """Compute the connectivity graph and edge attributes."""
        delaunay = tri.Triangulation(positions[:, 0], positions[:, 1])
        edges = self.extract_edges(delaunay.triangles)

        distances, displacements = self.compute_edge_metrics(edges, positions)
        valid_mask = distances < self.connectivity_radius

        edges = edges[valid_mask]
        distances = distances[valid_mask, None]
        displacements = displacements[valid_mask]

        edges, displacements, distances = self.make_graph_undirected(
            edges, displacements, distances
        )

        edge_index, edge_attr = self.format_edges_and_attributes(
            edges, displacements, distances
        )
        return edge_index, edge_attr

    @staticmethod
    def extract_edges(triangles):
        """Extract edges from triangles."""
        edges = [
            np.array(list(itertools.combinations(triangle, r=2)))[[1, 0, 2]]
            for triangle in triangles
        ]
        return np.concatenate(edges, axis=0)

    @staticmethod
    def compute_edge_metrics(edges, positions):
        """Compute displacements and distances for edges."""
        displacements = positions[edges[:, 0]] - positions[edges[:, 1]]
        distances = np.linalg.norm(displacements, axis=1)
        return distances, displacements

    @staticmethod
    def make_graph_undirected(edges, displacements, distances):
        """Ensure the graph is undirected by mirroring edges."""
        edges = np.concatenate([edges, edges[:, [1, 0]]], axis=0)
        displacements = np.concatenate([displacements, -displacements], axis=0)
        distances = np.concatenate([distances, distances], axis=0)

        # Remove duplicate edges
        unique_edges, indices = np.unique(edges, axis=0, return_index=True)
        return unique_edges, displacements[indices], distances[indices]

    def format_edges_and_attributes(self, edges, displacements, distances):
        """Format edges and their attributes for PyTorch Geometric."""
        edge_index = torch.tensor(edges.T, dtype=torch.long)
        edge_attr = (
            torch.cat(
                [
                    torch.tensor(displacements, dtype=torch.float),
                    torch.tensor(distances, dtype=torch.float),
                ],
                dim=1,
            )
            / self.connectivity_radius
        )
        return edge_index, edge_attr


class GraphBlinkingDataset:
    def __init__(
        self,
        training_clusters,
        training_backgrounds,
        dataset_size=512,
        cluster_count_range=(10, 20),
        cluster_sampling_range=(0.055, 0.062),
        cluster_std_sigma=0.003,
        num_cluster_additions=4,
        background_sampling_range=(0.035, 0.04),
        background_std_population=0.009,
        num_bkg_additions=4,
        connectivity_radius=1.0,
    ):
        self.dataset_size = dataset_size
        self.connectivity_radius = connectivity_radius

        self.training_clusters = training_clusters
        self.cluster_count_range = cluster_count_range
        self.cluster_sampling_range = cluster_sampling_range
        self.cluster_std_sigma = cluster_std_sigma
        self.num_cluster_additions = num_cluster_additions

        self.training_backgrounds = training_backgrounds
        self.background_sampling_range = background_sampling_range
        self.background_std_population = background_std_population
        self.num_background_additions = num_bkg_additions

        self.laplacian_embedding = AddLaplacianEigenvectorPE(
            5, attr_name="x", is_undirected=True
        )

        self.training_data = [
            self.populate_cluster(
                cluster, self.cluster_std_sigma, self.num_cluster_additions
            )
            for cluster in tqdm.tqdm_notebook(training_clusters)
        ]
        self.training_backgrounds = [
            self.populate_cluster(
                cluster, self.background_std_population, self.num_background_additions
            )
            for cluster in tqdm.tqdm_notebook(training_backgrounds)
        ]

    def __call__(self):
        return self.generate_dataset()

    def generate_dataset(self):
        """Generate a dataset of graph-structured data."""
        dataset = []

        for _ in tqdm.tqdm_notebook(
            range(self.dataset_size), desc="Generating dataset"
        ):
            noise_points, empty_spaces = self.generate_noise_points()
            clusters, deltas = self.generate_clusters(empty_spaces)

            positions = np.concatenate([clusters, noise_points], axis=0)
            data = self.create_data_object(positions, deltas)
            dataset.append(data)

        return dataset

    def generate_clusters(self, empty_spaces):
        """Generate clustered data points with random shifts and rotations."""
        num_clusters = np.random.randint(*self.cluster_count_range)

        random_shifts = np.array(empty_spaces)[
            np.random.choice(len(empty_spaces), num_clusters, replace=False)
        ]

        all_positions = []
        all_deltas = []
        for shift in random_shifts:
            cluster = self.sample_random_cluster()
            cluster = self.apply_random_rotation(cluster)
            shifted_positions = cluster + shift

            all_positions.append(shifted_positions)
            all_deltas.append(cluster)

        return np.concatenate(all_positions, axis=0), np.concatenate(all_deltas, axis=0)

    def populate_cluster(self, cluster, sigma, num_additions):
        cluster = cluster.copy()
        for i in range(num_additions):
            addition = np.random.normal(0, sigma, size=(cluster.shape[0], 2))
            addition = cluster[["x", "y"]].values + addition
            cluster = pd.concat([cluster, pd.DataFrame(addition, columns=["x", "y"])])
        return cluster

    def sample_random_cluster(self):
        """Sample a subset of the training data to represent a cluster."""
        cluster = self.training_data[
            np.random.randint(0, len(self.training_data))
        ].copy()
        return cluster.sample(frac=np.random.uniform(*self.cluster_sampling_range))[
            ["x", "y"]
        ].values

    def sample_random_background(self):
        """Sample a subset of the training data to represent a background."""
        cluster = self.training_backgrounds[
            np.random.randint(0, len(self.training_backgrounds))
        ].copy()
        return cluster.sample(frac=np.random.uniform(*self.background_sampling_range))[
            ["x", "y"]
        ].values

    @staticmethod
    def apply_random_rotation(points):
        """Apply a random rotation to a set of 2D points."""
        angle = np.random.uniform(0, 2 * np.pi)
        rotation_matrix = np.array(
            [
                [np.cos(angle), -np.sin(angle)],
                [np.sin(angle), np.cos(angle)],
            ]
        )
        return np.dot(points, rotation_matrix)

    def generate_noise_points(self):
        """Generate random noise points with optional flips and centering."""
        noise = self.sample_random_background()

        # Center the noise around the origin
        mean = np.mean(noise, axis=0)
        noise_centered = noise - mean

        # Random horizontal and vertical flips using a flip mask
        flip_mask = np.array(
            [-1 if np.random.rand() > 0.5 else 1, -1 if np.random.rand() > 0.5 else 1]
        )
        noise_flipped = noise_centered * flip_mask

        # Restore original mean
        noise_flipped = noise_flipped + mean

        return noise_flipped, self.find_empty_spaces(noise_flipped)

    def create_data_object(self, positions, deltas):
        """Create a `Data` object from positions and deltas."""
        data = Data()

        # Prepare labels
        noise_labels = np.zeros_like(positions[len(deltas) :])  # For noise
        data.y = (
            torch.tensor(
                np.concatenate([deltas, noise_labels], axis=0), dtype=torch.float
            )
            / self.connectivity_radius
        )

        # Add graph structure
        data.edge_index, data.edge_attr = self.compute_connectivity(positions)
        data.num_nodes = positions.shape[0]

        # Add positional and embedding features
        data = self.laplacian_embedding(data)
        data.x = torch.abs(data.x)
        data.position = torch.tensor(positions, dtype=torch.float)

        return data

    def compute_connectivity(self, positions):
        """Compute the connectivity graph and edge attributes."""
        delaunay = tri.Triangulation(positions[:, 0], positions[:, 1])
        edges = self.extract_edges(delaunay.triangles)

        distances, displacements = self.compute_edge_metrics(edges, positions)
        valid_mask = distances < self.connectivity_radius

        edges = edges[valid_mask]
        distances = distances[valid_mask, None]
        displacements = displacements[valid_mask]

        edges, displacements, distances = self.make_graph_undirected(
            edges, displacements, distances
        )

        edge_index, edge_attr = self.format_edges_and_attributes(
            edges, displacements, distances
        )
        return edge_index, edge_attr

    @staticmethod
    def extract_edges(triangles):
        """Extract edges from triangles."""
        edges = [
            np.array(list(itertools.combinations(triangle, r=2)))[[1, 0, 2]]
            for triangle in triangles
        ]
        return np.concatenate(edges, axis=0)

    @staticmethod
    def compute_edge_metrics(edges, positions):
        """Compute displacements and distances for edges."""
        displacements = positions[edges[:, 0]] - positions[edges[:, 1]]
        distances = np.linalg.norm(displacements, axis=1)
        return distances, displacements

    @staticmethod
    def make_graph_undirected(edges, displacements, distances):
        """Ensure the graph is undirected by mirroring edges."""
        edges = np.concatenate([edges, edges[:, [1, 0]]], axis=0)
        displacements = np.concatenate([displacements, -displacements], axis=0)
        distances = np.concatenate([distances, distances], axis=0)

        # Remove duplicate edges
        unique_edges, indices = np.unique(edges, axis=0, return_index=True)
        return unique_edges, displacements[indices], distances[indices]

    def format_edges_and_attributes(self, edges, displacements, distances):
        """Format edges and their attributes for PyTorch Geometric."""
        edge_index = torch.tensor(edges.T, dtype=torch.long)
        edge_attr = (
            torch.cat(
                [
                    torch.tensor(displacements, dtype=torch.float),
                    torch.tensor(distances, dtype=torch.float),
                ],
                dim=1,
            )
            / self.connectivity_radius
        )
        return edge_index, edge_attr

    def find_empty_spaces(self, point_cloud, resolution=0.02):
        """
        Finds empty grid cell centers in the 2D or 3D space covered by a point cloud.

        Parameters:
        - point_cloud: (N, D) array of points (D = 2 or 3).
        - resolution: float, size of the grid cells.

        Returns:
        - List of coordinates representing empty spaces (as centers of grid cells).
        """
        point_cloud = np.asarray(point_cloud)
        dims = point_cloud.shape[1]
        if dims not in [2, 3]:
            raise ValueError("Only 2D or 3D point clouds are supported.")

        min_bounds = np.floor(point_cloud.min(axis=0) / resolution) * resolution
        max_bounds = np.ceil(point_cloud.max(axis=0) / resolution) * resolution

        grid_axes = [
            np.arange(min_bounds[d], max_bounds[d], resolution) + resolution / 2
            for d in range(dims)
        ]
        mesh = np.meshgrid(*grid_axes, indexing="ij")
        grid_centers = np.stack([m.flatten() for m in mesh], axis=-1)

        point_indices = np.floor((point_cloud - min_bounds) / resolution).astype(int)
        unique_indices = set(map(tuple, point_indices))

        grid_indices = np.floor((grid_centers - min_bounds) / resolution).astype(int)
        grid_indices_set = set(map(tuple, grid_indices))

        empty_indices = grid_indices_set - unique_indices
        empty_coords = [
            min_bounds + resolution * (np.array(idx) + 0.5) for idx in empty_indices
        ]

        return empty_coords


class GraphBlinkingCroppedDataset:
    def __init__(
        self,
        training_clusters,
        training_backgrounds,
        dataset_size=512,
        cluster_count_range=(6, 10),
        cluster_sampling_range=(0.95, 1),
        background_Sampling_range=(0.9, 1),
        connectivity_radius=1.0,
        max_cropped_size=0.4,
    ):
        self.training_data = training_clusters
        self.training_backgrounds = training_backgrounds
        self.dataset_size = dataset_size
        self.cluster_count_range = cluster_count_range
        self.connectivity_radius = connectivity_radius
        self.background_Sampling_range = background_Sampling_range
        self.cluster_sampling_range = cluster_sampling_range

        self.max_cropped_size = max_cropped_size

        self.laplacian_embedding = AddLaplacianEigenvectorPE(
            5, attr_name="x", is_undirected=True
        )

    def __call__(self):
        return self.generate_dataset()

    def generate_dataset(self):
        """Generate a dataset of graph-structured data."""
        dataset = []

        for _ in tqdm.tqdm_notebook(
            range(self.dataset_size), desc="Generating dataset"
        ):
            noise_points = self.generate_noise_points()
            clusters, deltas = self.generate_clusters()

            positions = np.concatenate([clusters, noise_points], axis=0)
            data = self.create_data_object(positions, deltas)
            dataset.append(data)

        return dataset

    def generate_clusters(self):
        """Generate clustered data points with random shifts and rotations."""
        num_clusters = np.random.randint(*self.cluster_count_range)
        random_shifts = np.random.uniform(
            0.05, self.max_cropped_size, size=(num_clusters, 2)
        )

        all_positions = []
        all_deltas = []

        for shift in random_shifts:
            cluster = self.sample_random_cluster()
            cluster = self.apply_random_rotation(cluster)
            shifted_positions = cluster + shift

            all_positions.append(shifted_positions)
            all_deltas.append(cluster)

        return np.concatenate(all_positions, axis=0), np.concatenate(all_deltas, axis=0)

    def sample_random_cluster(self):
        """Sample a subset of the training data to represent a cluster."""
        cluster = self.training_data[
            np.random.randint(0, len(self.training_data))
        ].copy()
        return cluster.sample(frac=np.random.uniform(*self.cluster_sampling_range))[
            ["x", "y"]
        ].values

    def sample_random_background(self):
        """Sample a subset of the training data to represent a background."""
        cluster = self.training_backgrounds[
            np.random.randint(0, len(self.training_backgrounds))
        ].copy()
        return cluster[["x", "y"]].values

    def generate_noise_points(self):
        """Generate noise points within a specific cropped area."""
        noise = self.sample_random_background()

        init_x = np.random.rand() * (1 - self.max_cropped_size)
        init_y = np.random.rand() * (1 - self.max_cropped_size)
        where = (
            (noise[:, 0] > init_x)
            & (noise[:, 0] < init_x + self.max_cropped_size)
            & (noise[:, 1] > init_y)
            & (noise[:, 1] < init_y + self.max_cropped_size)
        )
        noise = noise[where]
        noise -= noise.min(axis=0)

        noise = noise[
            np.random.randint(
                0,
                len(noise),
                int(len(noise) * np.random.uniform(*self.background_Sampling_range)),
            )
        ]
        noise = self.apply_random_flip(noise)
        return noise

    def create_data_object(self, positions, deltas):
        """Create a `Data` object from positions and deltas."""
        data = Data()

        # Prepare labels
        noise_labels = np.zeros_like(positions[len(deltas) :])  # For noise
        data.y = (
            torch.tensor(
                np.concatenate([deltas, noise_labels], axis=0), dtype=torch.float
            )
            / self.connectivity_radius
        )

        # Add graph structure
        data.edge_index, data.edge_attr = self.compute_connectivity(positions)
        data.num_nodes = positions.shape[0]

        # Add positional and embedding features
        data = self.laplacian_embedding(data)
        data.x = torch.abs(data.x)
        data.position = torch.tensor(positions, dtype=torch.float)

        return data

    def compute_connectivity(self, positions):
        """Compute the connectivity graph and edge attributes."""
        delaunay = tri.Triangulation(positions[:, 0], positions[:, 1])
        edges = self.extract_edges(delaunay.triangles)

        distances, displacements = self.compute_edge_metrics(edges, positions)
        valid_mask = distances < self.connectivity_radius

        edges = edges[valid_mask]
        distances = distances[valid_mask, None]
        displacements = displacements[valid_mask]

        edges, displacements, distances = self.make_graph_undirected(
            edges, displacements, distances
        )

        edge_index, edge_attr = self.format_edges_and_attributes(
            edges, displacements, distances
        )
        return edge_index, edge_attr

    @staticmethod
    def extract_edges(triangles):
        """Extract edges from triangles."""
        edges = [
            np.array(list(itertools.combinations(triangle, r=2)))[[1, 0, 2]]
            for triangle in triangles
        ]
        return np.concatenate(edges, axis=0)

    @staticmethod
    def compute_edge_metrics(edges, positions):
        """Compute displacements and distances for edges."""
        displacements = positions[edges[:, 0]] - positions[edges[:, 1]]
        distances = np.linalg.norm(displacements, axis=1)
        return distances, displacements

    @staticmethod
    def make_graph_undirected(edges, displacements, distances):
        """Ensure the graph is undirected by mirroring edges."""
        edges = np.concatenate([edges, edges[:, [1, 0]]], axis=0)
        displacements = np.concatenate([displacements, -displacements], axis=0)
        distances = np.concatenate([distances, distances], axis=0)

        # Remove duplicate edges
        unique_edges, indices = np.unique(edges, axis=0, return_index=True)
        return unique_edges, displacements[indices], distances[indices]

    def format_edges_and_attributes(self, edges, displacements, distances):
        """Format edges and their attributes for PyTorch Geometric."""
        edge_index = torch.tensor(edges.T, dtype=torch.long)
        edge_attr = (
            torch.cat(
                [
                    torch.tensor(displacements, dtype=torch.float),
                    torch.tensor(distances, dtype=torch.float),
                ],
                dim=1,
            )
            / self.connectivity_radius
        )
        return edge_index, edge_attr

    @staticmethod
    def apply_random_rotation(points):
        """Apply a random rotation to a set of 2D points."""
        angle = np.random.uniform(0, 2 * np.pi)
        rotation_matrix = np.array(
            [
                [np.cos(angle), -np.sin(angle)],
                [np.sin(angle), np.cos(angle)],
            ]
        )
        return np.dot(points, rotation_matrix)

    @staticmethod
    def apply_random_flip(points):
        """Apply a random flip to a set of 2D points."""
        # Center the noise around the origin
        mean = np.mean(points, axis=0)
        points_centered = points - mean

        # Random horizontal and vertical flips using a flip mask
        flip_mask = np.array(
            [-1 if np.random.rand() > 0.5 else 1, -1 if np.random.rand() > 0.5 else 1]
        )
        points_flipped = points_centered * flip_mask
        return points_flipped + mean


def compute_test_graph(data, builder):
    # Extract position data from the test dataset
    position = data[["x", "y"]].values

    # Normalize the position coordinates to the range [0, 1]
    normalized_position = (position - position.min(axis=0)) / (
        position.max(axis=0) - position.min(axis=0)
    )

    # Initialize the graph data structure
    test_graph = Data(
        position=torch.tensor(
            normalized_position, dtype=torch.float32
        ),  # Node positions
        num_nodes=position.shape[0],  # Number of nodes
    )

    # Compute graph connectivity (edge indices and attributes) using the graph builder
    test_graph.edge_index, test_graph.edge_attr = builder.compute_connectivity(
        test_graph.position
    )

    # Compute the Laplacian embedding of the graph
    test_graph = builder.laplacian_embedding(test_graph)
    test_graph.x = torch.abs(test_graph.x)

    return test_graph.to("cuda" if torch.cuda.is_available() else "cpu")
