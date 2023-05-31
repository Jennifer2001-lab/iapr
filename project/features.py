import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.color import rgb2hsv, rgb2gray
from sklearn.decomposition import KernelPCA
from scipy.stats import kurtosis, skew

ORIENTATION = [(30, 30), (90, 90), (0, 0), (0, 90)]
COLORS = ["r", "g", "b", "k"]


def get_cut_image(img):
    return img[2:-2, 2:-2, :]


def get_borders_image(img):
    return (
        img[0, :, :],
        img[0, ::-1, :],
        img[-1, :, :],
        img[-1, ::-1, :],
        img[:, 0, :],
        img[::-1, 0, :],
        img[:, -1, :],
        img[::-1, -1, :],
    )


def correlate(img1, img2):
    return np.sum((img1 - np.mean(img1)) * (img2 - np.mean(img2))) / (
        np.std(img1) * np.std(img2) * np.size(img1)
    )


class Features:
    def __init__(self, pieces, dim_reduction=2):
        self.pieces = pieces
        self.pieces_ft = []
        self.dim_reduction = dim_reduction
        self.features = None
        self.features_PCA = None
        for piece in self.pieces:
            transform = np.fft.fft2(rgb2gray(piece))
            fshift = np.fft.fftshift(transform)
            # Keep only low frequencies to discard noise and take the log
            self.pieces_ft.append(np.log(np.abs(fshift[51:-50, 51:-50])))

    def find_correlation(self):
        correlations = []
        for img in self.pieces_ft:
            cor = np.array(
                [
                    max(correlate(b, img), correlate(np.rot90(b), img))
                    for b in self.pieces_ft
                ]
            )
            correlations.append(cor)
        self.correlations = np.array(correlations)

    def find_distances(self):
        distances = []
        for img in self.pieces_ft:
            dist = np.array(
                [
                    min(np.sum((b - img) ** 2), np.sum((np.rot90(b) - img) ** 2))
                    for b in self.pieces_ft
                ]
            )
            distances.append(dist)
        self.distances = np.array(distances)

    def find_distances_borders(self):
        if len(self.pieces) <= 1:
            self.distances_borders = np.array(len(self.pieces))
        cut_pieces = [get_cut_image(piece) for piece in self.pieces]
        dists = []  # Initialize an empty list to store distances

        # Loop over each cut piece
        for i, p1 in enumerate(cut_pieces):
            borders_p1 = get_borders_image(p1)
            dists_p1 = []  # Initialize an empty list to store distances for p1

            # Loop over each cut piece again to compare with p1
            for j, p2 in enumerate(cut_pieces):
                if i != j:  # Exclude the same piece comparison
                    borders_p2 = get_borders_image(p2)
                    dmin = np.inf  # Initialize the minimum distance to infinity

                    # Loop over each border of p2
                    for b in borders_p2:
                        # Compute the squared Euclidean distances and find the minimum
                        d = np.min(np.sum((b - borders_p1) ** 2, axis=1))
                        dmin = min(dmin, d)

                    if dmin != 0:  # Exclude zero distances
                        dists_p1.append(dmin)
            dists.append(np.sort(dists_p1)[0])
        self.distances_borders = np.array(dists).reshape(-1, 1)

    def find_features(self):
        features = np.array([_find_features(piece) for piece in self.pieces])
        self.find_distances()
        self.find_correlation()
        self.find_distances_borders()
        # Stack
        features = np.hstack(
            [features, self.distances, self.correlations, self.distances_borders]
        )
        # Normalize
        self.features = (features - np.mean(features, axis=0)) / np.maximum(
            1e-10, np.std(features, axis=0)
        )

    def find_features_PCA(self):
        self.find_features()
        pca = KernelPCA(n_components=self.dim_reduction, kernel="rbf")
        self.features_PCA = pca.fit_transform(self.features)

    def plot_features(self):
        fig = plt.figure()
        if self.features_PCA.shape[1] == 3:  # 3D plot
            for i, o in enumerate(ORIENTATION):
                ax = fig.add_subplot(2, 2, i + 1, projection="3d")
                ax.scatter(*self.features_PCA.T, c=COLORS[0], marker="x")
                ax.view_init(elev=o[0], azim=o[1])
        else:  # 2D plot
            ax = fig.add_subplot(1, 1, 1)
            ax.scatter(*self.features_PCA.T, c=COLORS[0], marker="x")
        fig.suptitle("PCA Features before classification")
        plt.tight_layout()
        plt.show()


def _find_features(img):
    # COLOR
    color = _color_features(img)

    # FOURIER
    fourier = _fourier_features(img)

    # EDGES
    canny_img = cv2.Canny(img, 50, 200)
    num_edges = np.count_nonzero(canny_img)

    return np.hstack(
        [
            color,
            fourier,
            num_edges,
        ]
    )


def _color_features(img):
    median = np.median(img, axis=(0, 1))
    std = np.std(img, axis=(0, 1))
    mean = np.mean(img, axis=(0, 1))
    kurtosis_c = kurtosis(img, axis=(0, 1))
    skew_c = skew(img, axis=(0, 1))
    img_hsv = rgb2hsv(img)
    median_hsv = np.median(img_hsv, axis=(0, 1))
    std_hsv = np.std(img_hsv, axis=(0, 1))
    mean_hsv = np.mean(img_hsv, axis=(0, 1))
    kurtosis_hsv = kurtosis(img_hsv, axis=(0, 1))
    skew_hsv = skew(img_hsv, axis=(0, 1))

    return np.hstack(
        [
            median,
            std,
            mean,
            kurtosis_c,
            skew_c,
            median_hsv,
            std_hsv,
            mean_hsv,
            kurtosis_hsv,
            skew_hsv,
        ]
    )


def _fourier_features(img):
    transform = np.fft.fft2(rgb2gray(img))
    ft = np.log(np.abs(np.fft.fftshift(transform)))[1:, 1:]  # [51:-50, 51:-50]

    ft_median = np.median(ft)
    ft_mean = np.mean(ft)
    ft_std = np.std(ft)
    ft_kurtosis = kurtosis(ft, axis=(0, 1))
    ft_skew = skew(ft, axis=(0, 1))

    binary_ft = ft > 4.5
    non_zero = np.count_nonzero(binary_ft)
    v = np.stack(np.where(binary_ft), axis=1)
    cov = np.cov(v.T)
    eigval, eigvec = np.linalg.eig(cov)

    idx = np.argsort(eigval)
    eigval, eigvec = eigval[idx], eigvec[idx]

    angle = np.arctan2(eigvec[0][1], eigvec[0][0]) % (np.pi / 2)

    k, l = np.indices((127, 127)) - 63
    m_00 = np.sum(binary_ft)
    m_20 = np.sum(k**2 * binary_ft)
    m_02 = np.sum(l**2 * binary_ft)
    M1 = m_02 + m_20

    return np.hstack(
        [
            ft_median,
            ft_mean,
            ft_std,
            ft_kurtosis,
            ft_skew,
            non_zero,
            eigval,
            angle,
            M1 / m_00,
        ]
    )
