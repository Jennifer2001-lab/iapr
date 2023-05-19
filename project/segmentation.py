import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndi
import cv2
import skimage

SHAPE = (2000, 2000)
PIECE_WIDTH = 128


class Segementation:
    def __init__(self, img):
        self.img = img

        self.hsv_img = skimage.color.rgb2hsv(self.img)
        self.hsv_imgs = {
            "Hue": self.hsv_img[:, :, 0],
            "Saturation": self.hsv_img[:, :, 1],
            "Value": self.hsv_img[:, :, 2],
        }
        self.hsv_edges = {}
        self.hsv_filled = {}
        self.hsv_blur = {}
        self.hsv_RoI = {}
        self.RoI_list = []
        self.mask = np.zeros(SHAPE)
        self.contours = []
        self.pieces = []
        plt.rcParams["image.cmap"] = "gray"

    def _find_edges(self):
        for key, gray_img in self.hsv_imgs.items():
            median_img = cv2.medianBlur((255 * gray_img).astype("uint8"), ksize=25)
            edges_sobel = skimage.filters.sobel(median_img)
            edges = cv2.Canny((edges_sobel * 255).astype("uint8"), 10, 100)
            self.hsv_edges[key] = edges
            print("Found edges", end="\r")

    def _fill_squares(self):
        self._find_edges()
        for key, edges in self.hsv_edges.items():
            dil = skimage.morphology.remove_small_holes(
                edges.astype(bool), PIECE_WIDTH**2
            )
            dil = skimage.morphology.dilation(dil, footprint=skimage.morphology.disk(5))
            dil = skimage.morphology.remove_small_holes(dil, PIECE_WIDTH**2)
            if (np.count_nonzero(dil) / dil.size) < 0.25:
                self.hsv_filled[key] = dil
            else:
                self.hsv_filled[key] = np.zeros(SHAPE, dtype=bool)
            print("Filled Squares  ", end="\r")

    def _blur(self):
        self._fill_squares()
        for key, filled in self.hsv_filled.items():
            self.hsv_blur[key] = ndi.gaussian_filter(filled.astype("float32"), sigma=40)
        print("Blured image      ", end="\r")

    def _find_peaks(self):
        self._blur()
        for key, blur_img in self.hsv_blur.items():
            centered = blur_img[1:-1, 1:-1]
            right = blur_img[1:-1, 2:]
            left = blur_img[1:-1, :-2]
            top = blur_img[:-2, 1:-1]
            bottom = blur_img[2:, 1:-1]
            img_peaks = (
                (centered > right)
                & (centered > left)
                & (centered > bottom)
                & (centered > top)
                & (centered > 0.6)
            )
            indx = np.where(img_peaks == 1)
            RoI = np.transpose(np.vstack((indx[0], indx[1])))
            if len(RoI) < 40:
                self.hsv_RoI[key] = RoI
                self.RoI_list.extend(RoI)
            else:
                self.hsv_RoI[key] = np.array([])
        print("Found peaks      ", end="\r")

    def _create_mask(self):
        self._find_peaks()
        mask_bkg = 2 * np.ones(SHAPE)
        mask_frg = np.zeros(SHAPE)
        indices = np.indices(SHAPE)

        for RoI in self.RoI_list:
            mask_frg[
                np.sqrt((indices[0] - RoI[0]) ** 2 + (indices[1] - RoI[1]) ** 2) < 80
            ] = 1

        self.mask = (mask_frg + mask_bkg).astype("uint8")
        print("Created mask      ", end="\r")

    def _grabCut(self):
        self._create_mask()
        backgroundModel = np.zeros((1, 65), np.float64)
        foregroundModel = np.zeros((1, 65), np.float64)

        cv2.grabCut(
            self.img,  # (255 * self.hsv_img).astype("uint8"),
            self.mask,
            rect=None,
            bgdModel=backgroundModel,
            fgdModel=foregroundModel,
            iterCount=2,
            mode=cv2.GC_INIT_WITH_MASK,
        )
        print("GrabCut finished", end="\r")

    def _clean_mask(self):
        self._grabCut()
        mask = np.where((self.mask == 2) | (self.mask == 0), 0, 1).astype(bool)
        mask = skimage.morphology.remove_small_holes(mask, 120**2)
        self.mask = skimage.morphology.remove_small_objects(mask, 120**2)
        print("Cleaned mask     ", end="\r")

    def _find_contours(self):
        self._clean_mask()
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
                122**2 < area < 130**2
                and 122 < perimeter / 4 < 130
                and 0.97 < area / (perimeter / 4) ** 2 < 1.03
            ):
                self.contours.append(cnt)
        print("Found Contours     ", end="\r")

    def find_pieces(self):
        self._find_contours()
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
            result = img_rot[
                round(h - PIECE_WIDTH) // 2 : -round(h - PIECE_WIDTH) // 2,
                round(w - PIECE_WIDTH) // 2 : -round(w - PIECE_WIDTH) // 2,
            ]
            # piece_filtered = cv2.medianBlur(result, ksize=3)
            self.pieces.append(result)
        print("Pieces detected :)", end="\r")

    # Visualization functions
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
        for (key, edges), ax in zip(self.hsv_edges.items(), axs.ravel()):
            ax.imshow(~edges)
            ax.set_title(key)
            ax.set_xticks([])
            ax.set_yticks([])

        fig.suptitle("Edges", fontsize=20)
        plt.tight_layout()
        plt.show()

    def plot_filled_squares(self):
        fig, axs = plt.subplots(1, 3, figsize=(15, 6))
        for (key, dil), ax in zip(self.hsv_filled.items(), axs.ravel()):
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
        for (key, blur), ax in zip(self.hsv_blur.items(), axs.ravel()):
            ax.imshow(1 - blur)
            ax.set_title(key)
            ax.set_xticks([])
            ax.set_yticks([])
            if not blur.any():
                ax.set_xlabel("Not used")
        fig.suptitle("Blured Binary Image", fontsize=20)
        plt.tight_layout()
        plt.show()

    def plot_RoI(self):
        fig, axs = plt.subplots(1, 3, figsize=(15, 6))
        for (key, peaks), ax in zip(self.hsv_RoI.items(), axs.ravel()):
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
        ax2.imshow(self.img, alpha=0.25)
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
