import numpy as np
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from sklearn.metrics import calinski_harabasz_score

MAX_OUTLIERS = 4
NUM_PIECES = [16, 12, 9]
NUM_PIECES_INVERSE = [16, 12, 9]
NUM_CLUSTER_INVERSE = [4, 3, 2]
NUM_PUZZLES = [2, 3]
COLORS = ["r", "g", "b", "k"]
ORIENTATION = [(30, 30), (90, 90), (0, 0), (0, 90)]


def valid_labeling(clusters_sizes):
    num_outliers_clusters = np.count_nonzero(clusters_sizes < MAX_OUTLIERS)
    num_total_outliers = np.sum(
        [count_outliers(count_label) for count_label in clusters_sizes]
    )
    if (
        (
            (MAX_OUTLIERS < clusters_sizes) & (clusters_sizes < min(NUM_PIECES_INVERSE))
        ).any()
        or num_outliers_clusters > 1
        or num_total_outliers > MAX_OUTLIERS
    ):
        return False
    return True


def count_outliers(cluster_size):
    for num_pieces in NUM_PIECES_INVERSE:
        if cluster_size - num_pieces >= 0:
            return cluster_size - num_pieces
    return cluster_size


class Classification:
    def __init__(self, features_PCA):
        self.features_PCA = features_PCA

    def classify(self, method=1):
        if method == 1:
            self.classify_1()
        elif method == 2:
            self.classify_2()
        else:
            self.classify_3()

    def classify_1(self):
        for i in NUM_CLUSTER_INVERSE:
            gm = GaussianMixture(n_components=i, n_init=10, init_params="k-means++")
            labels = gm.fit_predict(self.features_PCA.copy())
            clusters_sizes, _ = np.histogram(labels, bins=len(np.unique(labels)))
            if valid_labeling(clusters_sizes):  # Consider only valid clustering
                for label in np.unique(labels):
                    cluster_samples = self.features_PCA[labels == label]
                    cluster_size = clusters_sizes[label]
                    if len(cluster_samples) not in NUM_PIECES_INVERSE:
                        # Look for outliers in cluster
                        num_outliers = count_outliers(cluster_size)
                        centroid = gm.means_[label]
                        distances = np.sqrt(
                            np.sum((cluster_samples - centroid) ** 2, axis=1)
                        )
                        outliers_index = np.argsort(distances)[-num_outliers:]
                        new_labels = label * np.ones(len(cluster_samples))
                        new_labels[outliers_index] = -1
                        labels[labels == label] = new_labels

                self.labels = labels
                break

    def set_outliers(self):
        # Label outliers cluster to -1
        for label in np.unique(self.labels):
            if np.count_nonzero(self.labels == label) < 9:
                self.labels[self.labels == label] = -1

    def classify_2(self):
        max_calinski_harabasz_score = 0
        for i in NUM_PUZZLES:

            def gmm():
                gm = GaussianMixture(n_components=i, n_init=10, init_params="k-means++")
                labels = gm.fit_predict(self.features_PCA)
                for label in np.unique(labels):
                    cluster_samples = self.features_PCA[labels == label]
                    cluster_size = len(cluster_samples)
                    if (
                        cluster_size < min(NUM_PIECES)
                        or cluster_size - max(NUM_PIECES) > MAX_OUTLIERS
                    ):
                        return None  # Fit is not valid
                    possible_configurations = []  #
                    for n in NUM_PIECES:
                        if cluster_size >= n and cluster_size - n <= MAX_OUTLIERS:
                            possible_configurations.append((n, cluster_size - n))
                    # Example possible_configurations for cluster size = 13:
                    # (9,4) : 9 pieces, 4 outliers
                    # (12,1) : 12 pieces, 1 outliers
                    if len(possible_configurations) >= 2:
                        # Seperate cluster in two
                        gm_test = GaussianMixture(n_components=2, n_init=10)
                        labels_test = gm_test.fit_predict(cluster_samples)
                        num_c0 = list(labels_test).count(0)
                        num_c1 = list(labels_test).count(1)
                        # Check if fit is compatible with one configuration
                        if (
                            max(num_c0, num_c1),
                            min(num_c0, num_c1),
                        ) in possible_configurations:
                            num_outliers = min(num_c0, num_c1)
                        else:
                            if possible_configurations[1] == 0:
                                # If separation doesn't make sense maybe there isn't any outliers
                                num_outliers = 0
                            else:
                                # Number of outliers in cluster is impossible
                                return None
                    else:
                        num_outliers = possible_configurations[0][1]
                    if num_outliers != 0:
                        # Remove outliers in cluster
                        centroid = gm.means_[label]
                        distances = np.sqrt(
                            np.sum((cluster_samples - centroid) ** 2, axis=1)
                        )
                        outliers_index = np.argsort(distances)[-num_outliers:]
                        new_labels = label * np.ones(cluster_size)
                        new_labels[outliers_index] = -1
                        labels[labels == label] = new_labels
                if np.count_nonzero(labels == -1) > MAX_OUTLIERS:
                    return None  # Too many outliers, not a possible fit
                return labels

            labels = gmm()
            if np.any(labels):
                ch_score = calinski_harabasz_score(self.features_PCA, labels)
                print(
                    "Possible classification\n"
                    f"Number of pieces clusters: {np.unique(labels[labels!=-1])}",
                    "\n" f"Number of outliers: {np.count_nonzero(labels==-1)}",
                    "\n" f"Calinski Harabasz Score: {ch_score}",
                    "\n",
                )
                if ch_score > max_calinski_harabasz_score:  # Keep best fit
                    best_labels = labels
                    max_calinski_harabasz_score = ch_score
        print(
            f"Best classification\n"
            f"Number of pieces clusters: {np.unique(best_labels[best_labels!=-1])}",
            "\n" f"Number of outliers: {np.count_nonzero(best_labels==-1)}",
            "\n" f"Calinski Harabasz Score: {ch_score}",
            "\n",
        )
        self.labels = best_labels

    def classify_3(self):
        gm = [None] * 2
        bic_values = []

        for i in NUM_PUZZLES:
            gm = GaussianMixture(n_components=i, n_init=10, init_params="k-means++")
            labels = gm.fit_predict(self.features_PCA)

            for label in np.unique(labels):
                cluster_samples = self.features_PCA[labels == label]
                cluster_size = len(cluster_samples)

                for n in NUM_PIECES:
                    pieces_mask = np.ones(len(self.features_PCA), dtype=bool)

                    if (
                        (n == 9 and 9 < cluster_size <= 9 + MAX_OUTLIERS)
                        or (n == 12 and 12 < cluster_size <= 12 + MAX_OUTLIERS)
                        or (n == 16 and cluster_size > 16)
                    ):
                        centroid = gm.means_[label]
                        distances = np.sqrt(
                            np.sum((cluster_samples - centroid) ** 2, axis=1)
                        )
                        distances_sorted = np.sort(distances)
                        threshold = distances_sorted[n - 1]
                        outlier_mask = distances > threshold
                        pieces_mask[labels == label] &= ~outlier_mask

                    labels_filtered = labels.copy()
                    labels_filtered[~pieces_mask] = -1
                    features_pieces = self.features_PCA[pieces_mask]

                    num_pieces_filtered = []
                    for label_check in np.unique(labels):
                        num_pieces_filtered.append(
                            np.count_nonzero(labels_filtered == label_check)
                        )

                    if all(num in NUM_PIECES for num in num_pieces_filtered):
                        gm.fit(features_pieces)  # Check fit without outliers
                        bic_value = gm.bic(features_pieces)
                        bic_values.append((bic_value, labels_filtered))
                        print(
                            f"Model: {i+2}, Pieces: {n}, Label: {label}, BIC: {bic_value}"
                        )
        bic_values_only = np.array([x[0] for x in bic_values])
        print(bic_values_only)
        bic_min_index = np.argmin(bic_values_only)
        best_labeling = bic_values[bic_min_index][1]
        self.labels = best_labeling

    def plot_classification(self):
        fig = plt.figure()
        if self.features_PCA.shape[1] == 3:  # 3D plot
            for i, o in enumerate(ORIENTATION):
                ax = fig.add_subplot(2, 2, i + 1, projection="3d")
                for label in np.unique(self.labels):
                    ax.scatter(
                        *self.features_PCA[self.labels == label].T,
                        c=COLORS[label],
                        marker="x",
                        label=f"{label} ({np.count_nonzero(self.labels==label)})",
                    )
                    ax.view_init(elev=o[0], azim=o[1])
                ax.legend()
        else:  # 2D plot
            ax = fig.add_subplot(1, 1, 1)
            for label in np.unique(self.labels):
                ax.scatter(
                    *self.features_PCA[self.labels == label].T,
                    c=COLORS[label],
                    marker="x",
                    label=f"{label} ({np.count_nonzero(self.labels==label)})",
                )
            ax.legend()
        fig.suptitle("PCA Features classified")
        plt.tight_layout()
        plt.show()
