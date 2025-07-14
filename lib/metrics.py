import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull, Delaunay
from scipy.special import comb
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import (
    adjusted_rand_score,
    adjusted_mutual_info_score,
    confusion_matrix,
)
import matplotlib.pyplot as plt
import scipy.spatial as sp
import tqdm


def create_mask(hull_points, mask_dim, xmin, xmax, ymin, ymax):
    """
    Generates a binary mask from a convex hull.

    Parameters:
    - hull_points (array): Points forming the convex hull.
    - mask_dim (int): Dimension of the square mask (mask_dim x mask_dim).
    - xmin, xmax (float): Minimum and maximum x-coordinates of the bounding box.
    - ymin, ymax (float): Minimum and maximum y-coordinates of the bounding box.

    Returns:
    - mask (array): Binary mask indicating the area enclosed by the convex hull.
    """
    x = np.linspace(xmin, xmax, mask_dim)
    y = np.linspace(ymin, ymax, mask_dim)
    xx, yy = np.meshgrid(x, y)
    points = np.vstack((xx.flatten(), yy.flatten())).T

    # Delaunay triangulation for hull points
    delaunay = Delaunay(hull_points)
    mask = delaunay.find_simplex(points) >= 0
    return mask.reshape(xx.shape)


def compute_ARI(GTdata, class_data):
    """
    Calculates the Adjusted Rand Index (ARI) for comparing ground truth cluster assignments
    with predicted cluster assignments.

    Parameters:
    - GTdata (pd.DataFrame): Ground truth data containing the "index" column
                             representing cluster labels for each point.
    - class_data (list): A list of dictionaries, each containing "result" as the predicted
                         cluster labels for the corresponding test case.

    Returns:
    - ARI_values (array): A NumPy array containing the ARI score for each predicted clustering.
    """
    # Initialize an array to store ARI scores
    ARI_values = np.zeros(len(class_data))

    # Compute ARI for each predicted clustering
    for i, cluster_data in enumerate(class_data):
        ARI_values[i] = adjusted_rand_score(GTdata["index"], cluster_data["result"])

    return ARI_values


def compute_IoU(GTdata, class_data, mask_dim):
    """
    Calculates the Intersection over Union (IoU) between ground truth clusters and predicted clusters.

    Parameters:
    - GTdata (pd.DataFrame): Ground truth data containing "x", "y", and "index" columns.
    - class_data (list): List of dictionaries with predicted clustering results.
    - mask_dim (int): Dimension of the binary mask used for IoU calculations.

    Returns:
    - IoU_values (array): IoU values for each predicted cluster.
    """
    IoU_values = np.zeros(len(class_data))

    # Extract coordinates and cluster indices from ground truth data
    moleculeCoords = np.vstack((GTdata["x"], GTdata["y"])).T
    moleculeClusterIndex = GTdata["index"]

    # Bounding box dimensions
    xmin, xmax = np.min(moleculeCoords[:, 0]), np.max(moleculeCoords[:, 0])
    ymin, ymax = np.min(moleculeCoords[:, 1]), np.max(moleculeCoords[:, 1])

    # Identify unique ground truth clusters (ignoring background, index <= 0)
    unique_indices = np.unique(moleculeClusterIndex)
    unique_indices = unique_indices[unique_indices > 0]

    # Create ground truth masks
    GTMasks = []
    for u in unique_indices:
        cluster = moleculeCoords[moleculeClusterIndex == u]
        if cluster.shape[0] < 3:
            mask = np.zeros((mask_dim, mask_dim))  # Invalid cluster: no convex hull
        else:
            hull = ConvexHull(cluster)
            mask = create_mask(cluster[hull.vertices], mask_dim, xmin, xmax, ymin, ymax)
        GTMasks.append(mask)

    # Combine ground truth masks into a single binary mask
    flat_GT_mask = (np.sum(GTMasks, axis=0) > 0).astype(int)

    # Process predicted clusters
    for c, cluster_data in enumerate(class_data):
        clusterRes = cluster_data["result"]
        counts = np.bincount(clusterRes)
        valid_clusters = np.where(counts >= 3)[0]  # Ignore small clusters
        valid_clusters = valid_clusters[valid_clusters > 0]

        ResMasks = []
        for cluster in valid_clusters:
            cluster_points = moleculeCoords[clusterRes == cluster]
            hull = ConvexHull(cluster_points)
            mask = create_mask(
                cluster_points[hull.vertices], mask_dim, xmin, xmax, ymin, ymax
            )
            ResMasks.append(mask)

        if ResMasks:
            flat_Res_mask = (np.sum(ResMasks, axis=0) > 0).astype(int)

            # Calculate IoU
            sum_mask = flat_Res_mask + flat_GT_mask
            overlap = np.sum(sum_mask == 2)  # Intersection
            union = np.sum((sum_mask > 0).astype(int))  # Union
            IoU_values[c] = overlap / union if union > 0 else 0

    return IoU_values


def compute_ARI_clust(GTdata, class_data, class_exclude=[0]):
    """
    Compute Adjusted Rand Index (ARI) for clustering results.

    Parameters:
    - GTdata (pd.DataFrame): Ground truth data with a column "index" containing true labels.
    - class_data (list of dict): List of clustering results; each dict contains a "result" key with predicted labels.
    - class_exclude (list): List of labels to exclude from the analysis.

    Returns:
    - np.ndarray: Array of ARI values for each clustering result.
    """
    ARI_values = np.zeros(len(class_data))
    for i in range(len(class_data)):
        # Exclude specified classes
        mask = ~GTdata["index"].isin(class_exclude)
        filtered_GTdata_index = GTdata["index"][mask]
        filtered_class_result = class_data[i]["result"][mask]

        # Calculate ARI
        ARI_values[i] = adjusted_rand_score(
            filtered_GTdata_index, filtered_class_result
        )
    return ARI_values


def compute_AMI(GTdata, class_data):
    """
    Compute Adjusted Mutual Information (AMI) for clustering results.

    Parameters:
    - GTdata (pd.DataFrame): Ground truth data with a column "index" containing true labels.
    - class_data (list of dict): List of clustering results; each dict contains a "result" key with predicted labels.

    Returns:
    - np.ndarray: Array of AMI values for each clustering result.
    """
    AMI_values = np.zeros(len(class_data))
    for i in range(len(class_data)):
        AMI_values[i] = adjusted_mutual_info_score(
            GTdata["index"], class_data[i]["result"]
        )
    return AMI_values


def calculate_matching_table(true_labels, predicted_labels):
    """
    Calculate the matching table (confusion matrix) between true and predicted labels.

    Parameters:
    - true_labels (np.ndarray): Array of true labels.
    - predicted_labels (np.ndarray): Array of predicted labels.

    Returns:
    - np.ndarray: Confusion matrix.
    """
    return confusion_matrix(true_labels, predicted_labels)


def calculate_wi_vj_pi_qj_Awi_Avj(matching_table):
    """
    Compute metrics for analyzing clustering quality based on matching table.

    Parameters:
    - matching_table (np.ndarray): Confusion matrix.

    Returns:
    - tuple: Arrays of wi, vj, pi, qj, Awi, and Avj values.
    """
    n = np.sum(matching_table)  # Total number of objects
    N = n * (n - 1) / 2  # Total number of pairs
    ni_plus = np.sum(matching_table, axis=1)  # Row sums
    n_plus_j = np.sum(matching_table, axis=0)  # Column sums

    # Calculate P and Q
    P = np.sum([comb(ni, 2) for ni in ni_plus if ni >= 2])
    Q = np.sum([comb(nj, 2) for nj in n_plus_j if nj >= 2])

    # Calculate wi and vj
    wi = np.array(
        [
            (
                np.sum(
                    [
                        comb(matching_table[i, j], 2)
                        for j in range(matching_table.shape[1])
                        if matching_table[i, j] >= 2
                    ]
                )
                / comb(ni_plus[i], 2)
                if comb(ni_plus[i], 2) > 0
                else 0
            )
            for i in range(matching_table.shape[0])
        ]
    )

    vj = np.array(
        [
            (
                np.sum(
                    [
                        comb(matching_table[i, j], 2)
                        for i in range(matching_table.shape[0])
                        if matching_table[i, j] >= 2
                    ]
                )
                / comb(n_plus_j[j], 2)
                if comb(n_plus_j[j], 2) > 0
                else 0
            )
            for j in range(matching_table.shape[1])
        ]
    )

    # Calculate pi and qj
    pi = np.array(
        [
            comb(ni_plus[i], 2) / P if P > 0 else 0
            for i in range(matching_table.shape[0])
        ]
    )
    qj = np.array(
        [
            comb(n_plus_j[j], 2) / Q if Q > 0 else 0
            for j in range(matching_table.shape[1])
        ]
    )

    # Calculate Awi and Avj
    Awi = np.array(
        [
            (
                (
                    N
                    * np.sum(
                        [
                            comb(matching_table[i, j], 2)
                            for j in range(matching_table.shape[1])
                            if matching_table[i, j] >= 2
                        ]
                    )
                    - comb(ni_plus[i], 2) * Q
                )
                / (comb(ni_plus[i], 2) * (N - Q))
                if comb(ni_plus[i], 2) > 0
                else 0
            )
            for i in range(matching_table.shape[0])
        ]
    )

    Avj = np.array(
        [
            (
                (
                    N
                    * np.sum(
                        [
                            comb(matching_table[i, j], 2)
                            for i in range(matching_table.shape[0])
                            if matching_table[i, j] >= 2
                        ]
                    )
                    - comb(n_plus_j[j], 2) * P
                )
                / (comb(n_plus_j[j], 2) * (N - P))
                if comb(n_plus_j[j], 2) > 0
                else 0
            )
            for j in range(matching_table.shape[1])
        ]
    )

    return wi, vj, pi, qj, Awi, Avj


def compute_ARI_dagger(GTdata, class_data):
    """
    Compute ARI values using the refined "dagger" approach.

    Parameters:
    - GTdata (pd.DataFrame): Ground truth data with a column "index" containing true labels.
    - class_data (list of dict): List of clustering results; each dict contains a "result" key with predicted labels.

    Returns:
    - np.ndarray: Array of ARI dagger values for each clustering result.
    """
    ARI_values = np.zeros(len(class_data))
    for i in range(len(class_data)):
        matching_table = calculate_matching_table(
            GTdata["index"], class_data[i]["result"]
        )
        *_, Awi, Avj = calculate_wi_vj_pi_qj_Awi_Avj(matching_table)

        mean_Awi = np.mean(Awi[Awi != 0])
        mean_Avj = np.mean(Avj[Avj != 0])
        ARI = (
            2 * (mean_Awi * mean_Avj) / (mean_Awi + mean_Avj)
            if (mean_Awi + mean_Avj) > 0
            else 0
        )
        ARI_values[i] = ARI

    return ARI_values


def compute_stats(points, cluster):
    """
    Compute centroids, extents, and indices for each cluster.

    Parameters:
    - points (np.ndarray): Array of point coordinates.
    - cluster (np.ndarray): Array of cluster labels.

    Returns:
    - centroids (list): List of centroid coordinates for each cluster.
    - extents (list): List of extents (bounding box sizes) for each cluster.
    - indices (list): List of unique cluster indices.
    """
    uniques = np.unique(cluster)
    centroids, extents, indices = [], [], []

    for u in uniques:
        where = cluster == u
        centroids.append(np.mean(points[where], axis=0))
        extents.append(np.max(points[where], axis=0) - np.min(points[where], axis=0))
        indices.append(u)

    return centroids, extents, indices


def compute_matching(lc, pc, lext, pext, li, pi, beta=0.9):
    """
    Compute cluster matching based on centroid distance and extent similarity.

    Parameters:
    - lc (list): Centroids of the ground truth clusters.
    - pc (list): Centroids of the predicted clusters.
    - lext (list): Extents (bounding box sizes) of the ground truth clusters.
    - pext (list): Extents (bounding box sizes) of the predicted clusters.
    - li (list): Indices of the ground truth clusters.
    - pi (list): Indices of the predicted clusters.
    - beta (float): Weight for centroid distance vs extent similarity. Default is 0.9.

    Returns:
    - pairs (np.ndarray): Array of matched pairs of ground truth and predicted indices.
    - pairs_dist (np.ndarray): Array of centroid distances for the matched pairs.
    """
    costd = sp.distance.cdist(lc, pc, metric="euclidean")
    extents = (np.mean(lext, axis=-1) / 2)[:, None]
    costd[costd > extents] = 1e9  # Large penalty for large centroid distances.

    coste = sp.distance.cdist(lext, pext, metric="cityblock")
    cost = beta * costd + (1 - beta) * coste

    row_ind, col_ind = linear_sum_assignment(cost)
    li = np.array(li)
    pi = np.array(pi)
    pairs = np.stack([li[row_ind], pi[col_ind]]).T

    where = costd[row_ind, col_ind] < np.max(extents)
    pairs = pairs[where]
    pairs_dist = costd[row_ind, col_ind][where]

    return pairs, pairs_dist


def extract_clusters_from_df(data, index_column):
    """
    Extract clusters from the input DataFrame based on the specified index column.

    Parameters:
    - data (pd.DataFrame): Input DataFrame containing points and cluster labels.
    - index_column (str): Column name indicating cluster labels.

    Returns:
    - clusters (dict): Dictionary where keys are cluster indices and values are arrays of points.
    """
    clusters = {}
    for _, row in data.iterrows():
        x, y, cluster_idx = row["x"], row["y"], row[index_column]
        if cluster_idx not in clusters:
            clusters[cluster_idx] = []
        clusters[cluster_idx].append([x, y])

    for key in clusters:
        clusters[key] = np.array(clusters[key])

    return clusters


def pair_clusters(GTdata, class_data, class_exclude=[0], beta=0.9, plot=False):
    """
    Pair clusters based on centroid distance using the Hungarian algorithm.
    Optionally, visualize the clusters.

    Parameters:
    - GTdata (pd.DataFrame): Ground truth data with columns ["x", "y", "index"].
    - class_data (dict): Predicted results with a "result" key containing cluster labels.
    - class_exclude (list): List of labels to exclude. Default is [0].
    - beta (float): Weight for centroid distance vs extent similarity. Default is 0.9.
    - plot (bool): Whether to plot the results. Default is False.

    Returns:
    - TP (int): True positive count (matched clusters).
    - FP (int): False positive count (unmatched predicted clusters).
    - FN (int): False negative count (unmatched ground truth clusters).
    - RMSRE_N (float): Root mean square relative error in cluster sizes.
    - RMSE_centr (float): Root mean square error in centroid distances.
    """
    GTdata = GTdata.copy()
    GTdata["result"] = class_data["result"]

    mask_gt = ~GTdata["index"].isin(class_exclude)
    ground_truth_clusters = extract_clusters_from_df(GTdata[mask_gt], "index")

    mask_pred = ~GTdata["result"].isin(class_exclude)
    predicted_clusters = extract_clusters_from_df(GTdata[mask_pred], "result")

    gt_centroids, gt_extents, gt_indices = compute_stats(
        GTdata[mask_gt][["x", "y"]].values, GTdata[mask_gt]["index"].values
    )
    pred_centroids, pred_extents, pred_indices = compute_stats(
        GTdata[mask_pred][["x", "y"]].values, GTdata[mask_pred]["result"].values
    )

    pairs, pairs_dist = compute_matching(
        gt_centroids,
        pred_centroids,
        gt_extents,
        pred_extents,
        gt_indices,
        pred_indices,
        beta=beta,
    )

    TP, matched_gt, matched_pred, localization_errors, dist_centroids = (
        0,
        set(),
        set(),
        [],
        [],
    )

    for gt_idx, pred_idx in pairs:
        if gt_idx not in ground_truth_clusters or pred_idx not in predicted_clusters:
            continue

        n_gt = len(ground_truth_clusters[gt_idx])
        n_pred = len(predicted_clusters[pred_idx])
        localization_errors.append(((n_gt - n_pred) / n_gt) ** 2)
        dist_centroids.append(
            np.linalg.norm(
                np.array(gt_centroids[gt_indices.index(gt_idx)])
                - np.array(pred_centroids[pred_indices.index(pred_idx)])
            )
        )
        TP += 1
        matched_gt.add(gt_idx)
        matched_pred.add(pred_idx)

    FP = len(pred_indices) - len(matched_pred)
    FN = len(gt_indices) - len(matched_gt)
    RMSRE_N = np.sqrt(np.mean(localization_errors)) if localization_errors else 0
    RMSE_centr = np.sqrt(np.mean(dist_centroids)) if dist_centroids else 0

    if plot:
        fig, ax = plt.subplots(figsize=(10, 8))
        # Add plotting code here.
        plt.show()

    return TP, FP, FN, RMSRE_N, RMSE_centr


def compute_paired_metrics(GTdata, class_data, class_exclude=[0], beta=0.9, plot=False):
    """
    Compute metrics based on paired cluster matching for multiple predictions.

    Parameters:
    - GTdata (pd.DataFrame): Ground truth data with columns ["x", "y", "index"].
    - class_data (list): List of predicted results as dictionaries with a "result" key.
    - class_exclude (list): List of labels to exclude. Default is [0].
    - beta (float): Weight for centroid distance vs extent similarity. Default is 0.9.
    - plot (bool): Whether to plot the results. Default is False.

    Returns:
    - JI_values (np.ndarray): Jaccard index for each prediction.
    - RMSRE_N_values (np.ndarray): Root mean square relative error for each prediction.
    - RMSE_centr_values (np.ndarray): Root mean square error in centroid distances.
    """
    JI_values = np.zeros(len(class_data))
    RMSRE_N_values = np.zeros(len(class_data))
    RMSE_centr_values = np.zeros(len(class_data))

    for i in range(len(class_data)):
        TP, FP, FN, RMSRE_N, RMSE_centr = pair_clusters(
            GTdata, class_data[i], class_exclude=class_exclude, beta=beta, plot=plot
        )
        JI_values[i] = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0
        RMSRE_N_values[i] = RMSRE_N
        RMSE_centr_values[i] = RMSE_centr

    return JI_values, RMSRE_N_values, RMSE_centr_values


def calculate_metrics_for_experiments(df):
    """
    Calculate clustering metrics (IoU, ARI, AMI, etc.) for each experiment in the dataset.

    Parameters:
    - df (pd.DataFrame): A DataFrame containing the following columns:
        - "set" (str): Identifier for each experiment group.
        - "x", "y" (float): Coordinates of points in the experiment.
        - "index" (int): Ground truth cluster labels.
        - "clustering-MIRO" (int): Predicted cluster labels from the MIRO algorithm.
        - "clustering-DBSCAN" (int): Predicted cluster labels from the DBSCAN algorithm.

    Returns:
    - pd.DataFrame: A DataFrame containing the calculated metrics for each experiment and clustering method, with the following columns:
        - "experiment" (str): Identifier for the experiment.
        - "class_names" (str): Name of the clustering method ("MIRO" or "DBSCAN").
        - "IoU_values" (float): Intersection-over-Union (IoU) metric values.
        - "ARI_values" (float): Adjusted Rand Index (ARI) metric values.
        - "ARI_c_values" (float): Adjusted Rand Index (ARI) metric values computed for filtered data.
        - "ARI_dagger_values" (float): Adjusted Rand Index (ARI) using the refined "dagger" method.
        - "AMI_values" (float): Adjusted Mutual Information (AMI) metric values.
        - "JIc_values" (float): Jaccard index values based on cluster matching.
        - "RMSRE_N_values" (float): Root Mean Square Relative Error in cluster sizes.
        - "RMSE_centr_values" (float): Root Mean Square Error in centroid distances.
    """
    # Initialize a list to store results for all experiments
    all_results = []

    # Iterate over each unique experiment in the 'set' column
    for experiment in tqdm.tqdm_notebook(df["set"].unique()):
        # Filter the DataFrame for the current experiment
        experiment_data = df[df["set"] == experiment]

        # Define ground truth data (GTdata)
        GTdata_exp = experiment_data[["x", "y", "index"]]

        # Prepare class_data for MIRO and DBSCAN clustering results
        class_data = [
            {"result": experiment_data["clustering-MIRO"].values},
            {"result": experiment_data["clustering-DBSCAN"].values},
        ]

        # Calculate ARI values
        ARI_values = compute_ARI(GTdata_exp, class_data)
        ARI_clust_values = compute_ARI_clust(GTdata_exp, class_data)
        ARI_dagger_values = compute_ARI_dagger(GTdata_exp, class_data)

        # Compute AMI values
        AMI_values = compute_AMI(GTdata_exp, class_data)

        # Calculate IoU values
        ROIsize = (
            np.ceil(max(experiment_data["x"].max(), experiment_data["y"].max()) / 100)
            * 100
        )
        mask_dim = int(ROIsize + 400)
        IoU_values = compute_IoU(GTdata_exp, class_data, mask_dim)

        # Compute pairwise cluster matching and calculate metrics
        JIc_values, RMSRE_N_values, RMSE_centr_values = compute_paired_metrics(
            GTdata_exp, class_data, beta=0.9
        )

        # Store the results in a DataFrame for this experiment
        results_df = pd.DataFrame(
            {
                "experiment": experiment,
                "class_names": ["MIRO", "DBSCAN"],
                "IoU_values": IoU_values,
                "ARI_values": ARI_values,
                "ARI_c_values": ARI_clust_values,
                "ARI_dagger_values": ARI_dagger_values,
                "AMI_values": AMI_values,
                "JIc_values": JIc_values,
                "RMSRE_N_values": RMSRE_N_values,
                "RMSE_centr_values": RMSE_centr_values,
            }
        )

        # Append the results DataFrame to the list
        all_results.append(results_df)

    # Concatenate all experiment results into a single DataFrame
    final_results_df = pd.concat(all_results, ignore_index=True)

    return final_results_df
