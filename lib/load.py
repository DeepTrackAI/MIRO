import pandas as pd
import matplotlib.tri as tri
import numpy as np
import itertools
import tqdm

import torch
from torch_geometric.data import Data
from torch_geometric.transforms import AddLaplacianEigenvectorPE


class CreateGraphDataset:
    def __init__(
        self,
        training_cluster,
        dataset_size=512,
        noise_point_range=(150, 250),
        cluster_count_range=(10, 20),
        connectivity_radius=1.0,
    ):
        self.training_cluster = training_cluster
        self.dataset_size = dataset_size
        self.noise_point_range = noise_point_range
        self.cluster_count_range = cluster_count_range
        self.connectivity_radius = connectivity_radius

        self.laplacian_embedding = AddLaplacianEigenvectorPE(
            5, attr_name="x", is_undirected=True
        )

        self.training_data = self.populate_cluster(training_cluster)

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

    def populate_cluster(self, cluster, num_additions=4):
        cluster = cluster.copy()
        for i in range(num_additions):
            addition = np.random.normal(0, 0.003, size=(cluster.shape[0], 2))
            addition = cluster[["x", "y"]].values + addition
            cluster = pd.concat([cluster, pd.DataFrame(addition, columns=["x", "y"])])

        return cluster

    def sample_random_cluster(self):
        """Sample a subset of the training data to represent a cluster."""
        cluster = self.training_data.copy()
        return cluster.sample(frac=np.random.uniform(0.055, 0.062))[["x", "y"]].values

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
