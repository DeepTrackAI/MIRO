"""MIRO: Multimodal Integration through Relational Optimization

This module provides the MIRO framework for point cloud clustering, leveraging
advanced geometric deep learning techniques. MIRO transforms complex point
clouds into optimized representations, enabling more effective clustering
using traditional algorithms.

Based on the original MIRO paper by Pineda et al. [1], this implementation offers
easy-to-use methods for training the MIRO model and performing geometric-aware
clustering. It integrates recurrent graph neural networks to refine point
cloud data and enhance clustering accuracy.

[1] Pineda, Jesús, et al. "Spatial Clustering of Molecular Localizations with
    Graph Neural Networks." arXiv preprint arXiv:2412.00173 (2024).
"""

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Data
from sklearn.cluster import DBSCAN
from typing import Callable, Optional, List

from deeplay.applications import Application
from deeplay.external import Adam, Optimizer
from .miro import MIRO


class cMIRO(MIRO):
    """Point cloud clustering and shape classification using MIRO (Multimodal
    Integration through Relational Optimization).

    MIRO is a geometric deep learning framework that enhances clustering
    algorithms by transforming complex point clouds into an optimized structure
    amenable to conventional clustering methods. MIRO employs recurrent graph
    neural networks (rGNNs) to learn a transformation that squeezes localization
    belonging to the same cluster toward a common center, resulting in a compact
    representation of clusters within the point cloud.

    Parameters
    ----------
    num_outputs : int
        Dimensionality of the output features, representing a displacement
        vector in Cartesian space for each node. This vector points toward
        the center of each cluster.
    num_iterations : int
        Number of recurrent steps for the model to refine the point cloud.
    connectivity_radius : float
        Maximum distance between two nodes to consider them connected in the
        graph.
    model : nn.Module
        A model implementing the forward method. It should return a list of
        tensors of shape `(num_nodes, num_outputs)` representing the predicted
        displacement vectors for each node at each recurrent iteration. If not
        specified, a default model resembling the one from the original MIRO
        paper is used.
    nd_loss_weight : float
        Weight for the auxiliary loss that enforces preservation of pairwise
        distances between connected nodes. Default is 10.
    loss : torch.nn.Module
        Loss function for training. Default is `torch.nn.L1Loss`.
    optimizer : Optimizer
        Optimizer for training. Default is Adam with a learning rate of 1e-4.

    Returns
    -------
    forward : method
        Computes and returns the predicted displacement vectors for each node
        in the input graph. The output is a list of tensors representing the
        displacement vectors at each recurrent iteration.

    squeeze : method
        Applies the predicted displacement vectors from the last recurrent
        iteration (by default) to the input point cloud. This operation
        optimizes the point cloud for clustering by aligning nodes closer to
        their respective cluster centers.

    clustering : method
        Groups nodes into clusters using the DBSCAN algorithm, based on the
        predicted displacement vectors. Each node is assigned a cluster label,
        where -1 indicates background noise. Returns an array of cluster labels
        for the nodes.

    Example
    -------
    >>> # Predicts displacement vectors for each node in a point cloud at each
    >>> # recurrent iteration
    >>> displacement_vectors = model(test_graph)
    >>> print(type(displacement_vectors))
    <class 'list'>

    >>> # Applies the predicted displacement vectors to the input point cloud
    >>> squeezed = model.squeeze(test_graph)
    >>> print(squeezed.shape)
    (num_nodes, 2)

    >>> # Performs clustering using DBSCAN after MIRO squeezing
    >>> eps = 0.3  # Maximum distance for cluster connection
    >>> min_samples = 5  # Minimum points in a neighborhood for core points
    >>> clusters = model.clustering(test_graph, eps, min_samples)

    >>> # Output cluster labels
    >>> print(clusters)
    array([ 0,  0,  1,  1,  1, -1,  2,  2,  2, ...])
    # Nodes in cluster 0, 1, 2, etc.; -1 are outliers
    """

    num_classes: int
    connectivity_radius: float
    num_iterations: int
    model: nn.Module
    nd_loss_weight: float
    loss: torch.nn.Module
    metrics: list
    optimizer: Optimizer

    def __init__(
        self,
        num_classes: int = 3,
        num_iterations: int = 20,
        connectivity_radius: float = 1.0,
        model: Optional[nn.Module] = None,
        nd_loss_weight: float = 10,
        ce_loss_weight: float = 0.1,
        loss: torch.nn.Module = torch.nn.L1Loss(),
        ce_loss: torch.nn.Module = torch.nn.CrossEntropyLoss(),
        optimizer=None,
        **kwargs,
    ):

        super().__init__(
            num_outputs=2 + num_classes,
            num_iterations=num_iterations,
            connectivity_radius=connectivity_radius,
            model=model,
            nd_loss_weight=nd_loss_weight,
            loss=loss,
            optimizer=optimizer,
            **kwargs,
        )
        self.num_classes = num_classes
        self.ce_loss_weight = ce_loss_weight
        self.ce_loss = ce_loss

    def compute_loss(
        self,
        y_hat: List[torch.Tensor],
        y: torch.Tensor,
        edges: torch.Tensor,
        position: torch.Tensor,
    ) -> torch.Tensor:
        """Computes the total loss for the model."""
        total = sum(
            self.loss(p[:, :2], y.y)
            + self.nd_loss_weight * self.compute_nd_loss(p[:, :2], y.y, edges, position)
            + self.ce_loss_weight * self.ce_loss(p[:, 2:], y.y_class)
            for p in y_hat
        )
        return total / len(y_hat)

    def squeeze(
        self,
        x: Data,
        from_iter: int = -1,
        scaling: np.ndarray = np.array([1.0, 1.0]),
    ) -> np.ndarray:
        """Computes and applies the predicted displacement vectors to the
        input point cloud.

        Parameters
        ----------
         x : torch_geometric.data.Data
            Input graph data. It is expected to have the attributes:
            `x` (node features), `edge_index` (graph connectivity),
            `edge_attr` (edge features), and `positions` (node spatial coordinates).
        from_iter : int, optional
            Index of the recurrent iteration to be used as displacement vectors.
            Default is -1 (last iteration).
        scaling : np.ndarray, optional
            Scaling factors for each dimension. Default is [1.0, 1.0].

        Returns
        -------
        np.ndarray
            Squeezed point cloud with optimized cluster alignment.
        """
        with torch.no_grad():
            predictions = self(x)[from_iter].detach().cpu().numpy()
            predicted_displacements = predictions[:, :2]
            class_logits = predictions[:, 2:]

        positions = x.position.cpu().numpy()
        squeezed_positions = (
            positions - predicted_displacements * self.connectivity_radius
        )
        return np.concatenate(
            [
                squeezed_positions * scaling,
                np.argmax(class_logits, axis=1, keepdims=True),
            ],
            axis=1,
        )

    def classify(
        self,
        x: Data,
        from_iter: int = -1,
    ):
        """Classify nodes in the input graph using the predicted class logits.

        Parameters
        ----------
        x : torch_geometric.data.Data
            Input graph data.
        from_iter : int, optional
            Index of the recurrent iteration to be used for classification.
            Default is -1 (last iteration).

        Returns
        -------
        np.ndarray
            Predicted class labels for each node.
        """
        with torch.no_grad():
            predictions = self(x)[from_iter].detach().cpu().numpy()
            class_logits = predictions[:, 2:]
        return np.concatenate(
            [
                x.position.cpu().numpy(),
                np.argmax(class_logits, axis=1, keepdims=True),
            ],
            axis=1,
        )

    def clustering(
        self,
        x: Data,
        eps: float,
        min_samples: int,
        from_iter: int = -1,
        **kwargs,
    ) -> np.ndarray:
        """Perform clustering using DBSCAN after applying MIRO squeezing.

        Parameters
        ----------
        x : torch_geometric.data.Data
            Input graph data.
        eps : float
            The maximum distance between two samples for one to be considered
            as in the neighborhood of the other. This is not a maximum bound
            on the distances of points within a cluster. This is the most
            important DBSCAN parameter to choose appropriately for your data set
            and distance function.
        min_samples : int
            The number of samples (or total weight) in a neighborhood for a point
            to be considered as a core point. This includes the point itself.
        from_iter : int, optional
            Index of the recurrent iteration to be used as displacement vectors.
            Default is -1 (last iteration).

        Returns
        -------
        np.ndarray
            Cluster labels for each node. -1 indicates outliers.
        """
        squeezed, pclass = np.split(self.squeeze(x, from_iter=from_iter), [-1], axis=1)
        clusters = DBSCAN(eps=eps, min_samples=min_samples).fit(squeezed)
        rclass = self._refine_by_clustering(clusters.labels_, pclass.squeeze())
        return clusters.labels_, rclass

    def _refine_by_clustering(
        self,
        clusters: np.ndarray,
        predicted_classes: np.ndarray,
    ) -> np.ndarray:
        """
        Given a 1D array `clusters` of integer cluster IDs and a parallel 1D array
        `predicted_classes`, replace each preds[i] by the mode of its cluster.
        """
        predicted_classes = predicted_classes.astype(int)
        unique_clusters, inv = np.unique(clusters, return_inverse=True)
        K = unique_clusters.size

        modes = np.empty(K)
        for i in range(K):
            members = predicted_classes[inv == i]
            modes[i] = np.bincount(members).argmax()

        return modes[inv]
