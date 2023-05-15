import os
from PIL import Image
import numpy as np


def load_input_image(image_index, folder="train2", path="data_project"):
    filename = "train_{}.png".format(str(image_index).zfill(2))
    im = Image.open(os.path.join(path, folder, filename)).convert("RGB")
    return np.array(im)


def save_solution_puzzles(
    image_index,
    solved_puzzles,
    outliers,
    folder="train",
    path="data_project",
    group_id=0,
):
    path_solution = os.path.join(
        path, folder + "_solution_{}".format(str(group_id).zfill(2))
    )
    if not os.path.isdir(path_solution):
        os.mkdir(path_solution)

    print(path_solution)
    for i, puzzle in enumerate(solved_puzzles):
        filename = os.path.join(
            path_solution,
            "solution_{}_{}.png".format(str(image_index).zfill(2), str(i).zfill(2)),
        )
        Image.fromarray(puzzle).save(filename)

    for i, outlier in enumerate(outliers):
        filename = os.path.join(
            path_solution,
            "outlier_{}_{}.png".format(str(image_index).zfill(2), str(i).zfill(2)),
        )
        Image.fromarray(outlier).save(filename)


def rgb2gray(rgb):
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray
