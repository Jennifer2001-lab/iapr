import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndi
import cv2
import skimage

SHAPE = (2000, 2000)
PIECE_WIDTH = 128


class Segmentation:
    def __init__(self, img):
        self.img = img

        self.hsv_img = skimage.color.rgb2hsv(self.img)
        self.hsv_imgs = {
            "Hue": self.hsv_img[:, :, 0],
            "Saturation": self.hsv_img[:, :, 1],
            "Value": self.hsv_img[:, :, 2],
        }
        self.edges = {}
        self.closed_edges = {}
        self.blurred_edges = {}
        self.centers_pieces = {}
        self.centers_list = []
        self.mask = np.zeros(SHAPE)
        self.contours = []
        self.pieces = []
        plt.rcParams["image.cmap"] = "gray"

    def segment(self):
        self._find_edges()
        self._close_edges()
        self._blur()
        self._find_centers()
        self._create_mask()
        self._grabCut()
        self._clean_mask()
        self._find_contours()
        self._find_pieces()

    def _find_edges(self):
        for key, gray_img in self.hsv_imgs.items():
            median_img = cv2.medianBlur((255 * gray_img).astype("uint8"), ksize=25)
            edges_sobel = skimage.filters.sobel(median_img)
            edges = cv2.Canny((edges_sobel * 255).astype("uint8"), 10, 100)
            self.edges[key] = edges
            print("Found edges         ", end="\r")

    def _close_edges(self):
        for key, edges in self.edges.items():
            dil = skimage.morphology.remove_small_holes(
                edges.astype(bool), PIECE_WIDTH**2
            )
            dil = skimage.morphology.dilation(dil, footprint=skimage.morphology.disk(5))
            dil = skimage.morphology.remove_small_holes(dil, PIECE_WIDTH**2)
            if (np.count_nonzero(dil) / dil.size) < 0.25:
                self.closed_edges[key] = dil
            else:
                self.closed_edges[key] = np.zeros(SHAPE, dtype=bool)
            print("Filled Squares       ", end="\r")

    def _blur(self):
        for key, filled in self.closed_edges.items():
            self.blurred_edges[key] = ndi.gaussian_filter(
                filled.astype("float32"), sigma=40
            )
        print("Blured image          ", end="\r")

    def _find_centers(self):
        for key, blur_img in self.blurred_edges.items():
            centered = blur_img[1:-1, 1:-1]
            right = blur_img[1:-1, 2:]
            left = blur_img[1:-1, :-2]
            top = blur_img[:-2, 1:-1]
            bottom = blur_img[2:, 1:-1]
            local_max = (
                (centered > right)
                & (centered > left)
                & (centered > bottom)
                & (centered > top)
                & (centered > 0.6)
            )
            indx = np.where(local_max == 1)
            centers = np.transpose(np.vstack((indx[0], indx[1])))
            if len(centers) < 40:
                self.centers_pieces[key] = centers
                self.centers_list.extend(centers)
            else:
                self.centers_pieces[key] = np.array([])
        print("Found peaks          ", end="\r")

    def _create_mask(self):
        mask_bkg = 2 * np.ones(SHAPE)
        mask_frg = np.zeros(SHAPE)
        indices = np.indices(SHAPE)

        for centers in self.centers_list:
            mask_frg[
                np.sqrt((indices[0] - centers[0]) ** 2 + (indices[1] - centers[1]) ** 2)
                < 80
            ] = 1

        self.mask = (mask_frg + mask_bkg).astype("uint8")
        print("Created mask          ", end="\r")

    def _grabCut(self):
        backgroundModel = np.zeros((1, 65), np.float64)
        foregroundModel = np.zeros((1, 65), np.float64)

        cv2.grabCut(
            self.img,
            self.mask,
            rect=None,
            bgdModel=backgroundModel,
            fgdModel=foregroundModel,
            iterCount=2,  # Can be tuned
            mode=cv2.GC_INIT_WITH_MASK,
        )
        print("GrabCut finished    ", end="\r")

    def _clean_mask(self):
        # Remove noise
        mask = np.where((self.mask == 2) | (self.mask == 0), 0, 1).astype(bool)
        mask = skimage.morphology.remove_small_holes(mask, 120**2)
        self.mask = skimage.morphology.remove_small_objects(mask, 120**2)
        print("Cleaned mask        ", end="\r")

    def _find_contours(self):
        contours, _ = cv2.findContours(
            image=(255 * self.mask).astype("uint8"),
            mode=cv2.RETR_TREE,
            method=cv2.CHAIN_APPROX_SIMPLE,
        )
        for contour in contours:
            cnt = cv2.approxPolyDP(contour, cv2.arcLength(contour, True) / 8, True)
            perimeter = cv2.arcLength(cnt, True)
            area = cv2.contourArea(cnt)
            if (
                120**2 < area < 130**2
                and 120 < perimeter / 4 < 130
                and 0.97 < area / (perimeter / 4) ** 2 < 1.03
            ):  # Check if contour is valid
                self.contours.append(cnt)
        mask = np.zeros(self.mask.shape)
        cv2.drawContours(mask, self.contours, -1, 255, cv2.FILLED)
        self.mask = (mask / 255).astype(bool)
        print("Found Contours      ", end="\r")

    def _find_pieces(self):
        for contour in self.contours:
            rect = cv2.minAreaRect(contour)
            ((x, y), _, angle) = rect
            d = np.min(
                [
                    PIECE_WIDTH,
                    round(y),
                    round(x),
                    SHAPE[0] - round(y),
                    SHAPE[1] - round(x),
                ]
            )
            img_crop = self.img[
                round(y) - d : round(y) + d, round(x) - d : round(x) + d
            ]
            img_rot = ndi.rotate(img_crop, angle)
            (h, w) = img_rot.shape[:-1]
            piece = img_rot[
                round(h - PIECE_WIDTH) // 2 : -round(h - PIECE_WIDTH) // 2,
                round(w - PIECE_WIDTH) // 2 : -round(w - PIECE_WIDTH) // 2,
            ]
            self.pieces.append(piece)
        print("Pieces detected :) ", end="\r")

    # PLOTING FUNCTIONS
    def plot_hsv(self):
        fig, axs = plt.subplots(1, 3, figsize=(15, 6))
        for (key, img), ax in zip(self.hsv_imgs.items(), axs.ravel()):
            ax.imshow(img)
            ax.set_title(key)
            ax.set_xticks([])
            ax.set_yticks([])

        fig.suptitle("HSV Images", fontsize=20)
        plt.tight_layout()
        plt.show()

    def plot_edges(self):
        fig, axs = plt.subplots(1, 3, figsize=(15, 6))
        for (key, edges), ax in zip(self.edges.items(), axs.ravel()):
            ax.imshow(~edges)
            ax.set_title(key)
            ax.set_xticks([])
            ax.set_yticks([])

        fig.suptitle("Edges", fontsize=20)
        plt.tight_layout()
        plt.show()

    def plot_closed_edges(self):
        fig, axs = plt.subplots(1, 3, figsize=(15, 6))
        for (key, dil), ax in zip(self.closed_edges.items(), axs.ravel()):
            ax.imshow(~dil)
            ax.set_title(key)
            ax.set_xticks([])
            ax.set_yticks([])
            if not dil.any():
                ax.set_xlabel("Not used")
        fig.suptitle("Filled Squares", fontsize=20)
        plt.tight_layout()
        plt.show()

    def plot_blur(self):
        fig, axs = plt.subplots(1, 3, figsize=(15, 6))
        for (key, blur), ax in zip(self.blurred_edges.items(), axs.ravel()):
            ax.imshow(1 - blur)
            ax.set_title(key)
            ax.set_xticks([])
            ax.set_yticks([])
            if not blur.any():
                ax.set_xlabel("Not used")
        fig.suptitle("Blured Binary Image", fontsize=20)
        plt.tight_layout()
        plt.show()

    def plot_centers(self):
        fig, axs = plt.subplots(1, 3, figsize=(15, 6))
        for (key, peaks), ax in zip(self.centers_pieces.items(), axs.ravel()):
            ax.imshow(self.img, cmap="gray", alpha=0.5)
            ax.set_title(key)
            if peaks.any():
                ax.scatter(peaks[:, 1], peaks[:, 0], marker="x", color="k")
            ax.set_xticks([])
            ax.set_yticks([])
        fig.suptitle("Probable Pieces Centers", fontsize=20)
        plt.tight_layout()
        plt.show()

    def plot_mask(self):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

        ax1.imshow(self.mask)
        ax1.set_xticks([])
        ax1.set_yticks([])

        image_foreground = self.img * self.mask[:, :, np.newaxis]
        ax2.imshow(image_foreground)
        ax2.imshow(self.img, alpha=0.5, interpolation="nearest")
        ax2.set_xticks([])
        ax2.set_yticks([])

        fig.suptitle("Pieces Positions")
        plt.tight_layout()
        plt.show()

    def plot_contours(
        self,
    ):
        fig, ax = plt.subplots()
        ax.imshow(self.img, alpha=0.5)
        for contour in self.contours:
            contour = np.vstack((contour, contour[0][np.newaxis, :, :]))
            ax.plot(contour[:, :, 0], contour[:, :, 1], "-.k")
            ax.set_xticks([])
            ax.set_yticks([])
        fig.suptitle("Contours")
        plt.tight_layout()
        plt.show()

    def plot_pieces(self):
        num_pieces = len(self.pieces)

        fig, axs = plt.subplots(6, int(np.ceil(num_pieces / 6)))
        for i, (piece, ax) in enumerate(zip(self.pieces, axs.ravel())):
            ax.imshow(piece)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel(i, fontsize=8)
        for ax in axs.ravel()[num_pieces:]:
            ax.axis("off")
        fig.suptitle(f"{num_pieces} Detected Pieces")
        plt.tight_layout()
        plt.show()

    def plot_img(self):
        fig, ax = plt.subplots()
        ax.imshow(self.img)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.suptitle("Image")
        plt.tight_layout()
        plt.show()
