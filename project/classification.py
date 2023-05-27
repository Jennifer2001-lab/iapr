import numpy as np
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from sklearn.metrics import calinski_harabasz_score

MAX_OUTLIERS = 4
NUM_PIECES_INVERSE = [16, 12, 9]
NUM_CLUSTER_INVERSE = [4, 3, 2]
COLORS = ["r", "g", "b", "k"]
ORIENTATION = [(30, 30), (90, 90), (0, 0), (0, 90)]


class Classification:
    def __init__(self, features_PCA):
        self.features_PCA = features_PCA

    def classify(self, method=1):
        if method == 1:
            self.classify_1_()
        elif method == 2:
            self.classify_2_()
        else:
            self.classify_3_()

    def classify_1_(self):
        for i in NUM_CLUSTER_INVERSE:
            gm = GaussianMixture(n_components=i, n_init=10, init_params="k-means++")
            labels = gm.fit_predict(self.features_PCA.copy())
            counts, _ = np.histogram(labels, bins=len(np.unique(labels)))
            if valid_labels_(counts):
                for label in np.unique(labels):
                    cluster_samples = self.features_PCA[labels == label]
                    if len(cluster_samples) not in NUM_PIECES_INVERSE:
                        num_outliers = count_outliers_(counts[label])
                        # proba = gm.score_samples(cluster_samples)
                        # outliers_index = np.argsort(proba)[:num_outliers]
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
        for label in np.unique(self.labels):
            if np.count_nonzero(self.labels == label) < 9:
                self.labels[self.labels == label] = -1

    def init_plot(self):
        fig = plt.figure(figsize=(10, 10))
        for i, o in enumerate(ORIENTATION):
            ax = fig.add_subplot(2, 2, i + 1, projection="3d")
            ax.scatter(*self.features_PCA.T, c=COLORS[0], marker="x")
            ax.view_init(elev=o[0], azim=o[1])
        fig.suptitle("PCA Features before classification")
        plt.tight_layout()
        plt.show()

    def classified_plot(self):
        fig = plt.figure(figsize=(10, 10))
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
        fig.suptitle("PCA Features classified")
        plt.tight_layout()
        plt.show()

    def classify_2_(self):
        max_calinski_harabasz_score = 0

        for i in NUM_PUZZLES:

            def gmm():
                gm = GaussianMixture(n_components=i, n_init=10, init_params="k-means++")
                labels = gm.fit_predict(self.features_PCA)
                for label in np.unique(labels):
                    cluster_samples = self.features_PCA[labels == label]
                    cluster_num_pieces = len(cluster_samples)
                    if (
                        cluster_num_pieces < min(NUM_PIECES)
                        or cluster_num_pieces - max(NUM_PIECES) > MAX_OUTLIERS
                    ):
                        return None

                    possible_num_pieces = []
                    for n in NUM_PIECES:
                        if (
                            cluster_num_pieces >= n
                            and cluster_num_pieces - n <= MAX_OUTLIERS
                        ):
                            possible_num_pieces.append((n, cluster_num_pieces - n))
                            # par exemple (9,2) 9 pieces, 2 outliers

                    if len(possible_num_pieces) >= 2:
                        gm_test = GaussianMixture(n_components=2, n_init=10)
                        labels_test = gm_test.fit_predict(cluster_samples)
                        num_c0 = list(labels_test).count(0)
                        num_c1 = list(labels_test).count(1)
                        if (
                            max(num_c0, num_c1),
                            min(num_c0, num_c1),
                        ) in possible_num_pieces:
                            num_outliers = min(num_c0, num_c1)
                        else:
                            if possible_num_pieces[1] == 0:
                                num_outliers = 0
                            else:
                                return None

                    else:
                        num_outliers = possible_num_pieces[0][1]

                    # print(cluster_num_pieces - num_outliers, num_outliers)
                    if num_outliers != 0:
                        centroid = gm.means_[label]
                        distances = np.sqrt(
                            np.sum((cluster_samples - centroid) ** 2, axis=1)
                        )
                        outliers_index = np.argsort(distances)[-num_outliers:]
                        new_labels = label * np.ones(cluster_num_pieces)
                        new_labels[outliers_index] = -1
                        labels[labels == label] = new_labels
                if np.count_nonzero(labels == -1) > MAX_OUTLIERS:
                    return None
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
                if ch_score > max_calinski_harabasz_score:
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

    def classify_3_(self):
        gm = [None] * 2
        labels = [None] * 2
        aic_values = []

        for i in range(2):
            gm[i] = GaussianMixture(
                n_components=i + 2, n_init=10, init_params="k-means++"
            )
            labels[i] = gm[i].fit_predict(self.features_PCA)

            features_filtered = self.features_PCA.copy()

            for label in np.unique(labels[i]):
                current_label_mask = labels[i] == label
                cluster_samples = self.features_PCA[current_label_mask]
                num_pieces = len(cluster_samples)

                for n in NUM_PIECES:
                    valid_samples_mask = np.ones(len(self.features_PCA), dtype=bool)

                    if (
                        (n == 9 and 9 < num_pieces <= 12)
                        or (n == 12 and 12 < num_pieces <= 16)
                        or (n == 16 and num_pieces > 16)
                    ):
                        centroid = gm[i].means_[label]
                        distances = np.sqrt(
                            np.sum((cluster_samples - centroid) ** 2, axis=1)
                        )
                        distances_sorted = np.sort(distances)
                        threshold = distances_sorted[n - 1]
                        outlier_mask = distances > threshold
                        outlier_samples_mask = outlier_mask
                        valid_samples_mask[current_label_mask] &= ~outlier_samples_mask

                    labels_filtered = labels[i].copy()
                    labels_filtered[~valid_samples_mask] = -1
                    features_filtered = self.features_PCA[valid_samples_mask]

                    num_pieces_filtered = []
                    for label_check in np.unique(labels[i]):
                        num_pieces_filtered.append(
                            np.count_nonzero(labels_filtered == label_check)
                        )

                    if all(num in NUM_PIECES for num in num_pieces_filtered):
                        if len(features_filtered) >= i + 2:
                            gm[i].fit(features_filtered)
                            aic_value = gm[i].bic(features_filtered)
                            aic_values.append((aic_value, labels_filtered))
                            print(
                                f"Model: {i+2}, Pieces: {n}, Label: {label}, AIC: {aic_value}"
                            )
                        else:
                            print(
                                f"Model: {i+2}, Pieces: {n}, Label: {label}, AIC: Not enough samples"
                            )
        aic_values_only = np.array([x[0] for x in aic_values])
        print(aic_values_only)
        aic_min_index = np.argmin(aic_values_only)
        filtered_labels = aic_values[aic_min_index][1]

        self.labels = filtered_labels


def valid_labels_(counts):
    num_outliers_clusters = np.count_nonzero(counts < MAX_OUTLIERS)
    num_total_outliers = np.sum(
        [count_outliers_(count_label) for count_label in counts]
    )
    if (
        ((MAX_OUTLIERS < counts) & (counts < min(NUM_PIECES_INVERSE))).any()
        or num_outliers_clusters > 1
        or num_total_outliers > MAX_OUTLIERS
    ):
        return False
    return True


def count_outliers_(count_label):
    for num_pieces in NUM_PIECES_INVERSE:
        if count_label - num_pieces >= 0:
            return count_label - num_pieces
    return count_label


NUM_PIECES = [16, 12, 9]
NUM_PUZZLES = [2, 3]
MAX_OUTLIERS = 4
