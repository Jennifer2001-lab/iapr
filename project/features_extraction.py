import numpy as np
import cv2
from skimage.color import rgb2hsv, rgb2gray
from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import normalize
from scipy.stats import kurtosis, skew


class FeaturesExtraction:
    def __init__(self, pieces):
        self.pieces = pieces
        self.pieces_ft = []
        for piece in self.pieces:
            transform = np.fft.fft2(rgb2gray(piece))
            fshift = np.fft.fftshift(transform)
            self.pieces_ft.append(np.log(np.abs(fshift[51:-50, 51:-50])))

    def find_correlation_(self):
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

    def find_distances_(self):
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

    def find_features_(self):
        features = np.array([find_features_(piece) for piece in self.pieces])
        self.find_distances_()
        self.find_correlation_()
        features = np.hstack([features, self.distances, self.correlations])
        self.features = (features - np.mean(features, axis=0)) / np.maximum(
            1e-10, np.std(features, axis=0)
        )

    def find_features_PCA(self, n_components=2):
        self.find_features_()
        pca = KernelPCA(n_components=n_components, kernel="rbf")
        self.features_PCA = pca.fit_transform(self.features)


def find_features_(img):
    # COLOR
    color = color_features_(img)

    # FOURIER
    fourier = fourier_features_(img)

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


def color_features_(img):
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


def fourier_features_(img):
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
    m_00 = np.sum(ft)
    m_20 = np.sum(k**2 * ft)
    m_02 = np.sum(l**2 * ft)
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


def correlate(img1, img2):
    return np.sum((img1 - np.mean(img1)) * (img2 - np.mean(img2))) / (
        np.std(img1) * np.std(img2) * np.size(img1)
    )
