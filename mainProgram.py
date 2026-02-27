# -*- coding: utf-8 -*-
"""
Created on Mon Feb 23 14:43:14 2026

@author: karl
"""

import RulerInference
import FindMeasureKeypoints
import measureLength
import numpy as np
import re
import os
import pandas as pd
from wildlife_datasets import datasets
from wildlife_datasets.datasets import WildlifeDataset
from ultralytics import YOLO


def restrict(data, folders, idx):
    data, folders = data[idx], folders[idx]
    while True:
        max_col = np.max(folders.columns)
        if all(folders[max_col].isnull()):
            folders = folders.drop(max_col, axis=1)
        else:
            break
    return data, folders


def get_name(x):
    x_splits = x.split("_")
    if x.startswith("IMG_"):
        if len(x_splits) >= 3:
            return x_splits[2].split(".")[0]
    else:
        if len(x_splits) >= 2:
            return x_splits[1].split(".")[0]
    return None


class NewtsKent(WildlifeDataset):
    def create_catalogue(self, load_segmentation=False):
        data = datasets.utils.find_images(self.root)
        folders = data["path"].str.split(os.path.sep, expand=True)
        # if (
        #     folders[1].nunique() != 1
        #     and folders[1].iloc[0] != "Identification"
        # ):
        #     raise ValueError("Structure wrong")
        # idx = folders[3].isnull()
        # data, folders = restrict(data, folders, idx)

        data["identity"] = data["file"].apply(get_name)

        idx = ~folders[2].isnull()
        data, folders = restrict(data, folders, idx)
        idx = ~folders[2].apply(lambda x: x.startswith("Duplicated"))
        data, folders = restrict(data, folders, idx)
        # TODO: no idea what to do with these

        # TODO: possibly removing too many. now better than keeping bad
        idx = ~data["identity"].isnull()
        data, folders = restrict(data, folders, idx)
        # TODO: removing juveniles and other. will not work when 10k+ individuals are there
        idx = data["identity"].apply(len) <= 5
        data, folders = restrict(data, folders, idx)

        data["path"] = data["path"] + os.path.sep + data["file"]
        data["image_id"] = datasets.utils.create_id(
            data["path"].apply(lambda x: x.replace(os.path.sep, "/"))
        ).astype(str)
        # data["year"] = folders[0].apply(lambda x: int(x[:4]))
        data = data.drop("file", axis=1)

        if load_segmentation:
            cols = ["bbox_x", "bbox_y", "bbox_w", "bbox_h"]
            segmentation = pd.read_csv(f"{self.root}/segmentation.csv")
            data = pd.merge(data, segmentation, on="image_id", how="outer")
            data["bbox"] = list(data[cols].to_numpy())
            data["segmentation"] = data["segmentation"].apply(
                lambda x: eval(x)
            )
            data = data.drop(cols, axis=1)
            data = data.reset_index(drop=True)

        return self.finalize_catalogue(data)


base = "C:\\home\\programming\\neuralNewtwork"
root = os.path.join(base, "datasets", "crestedNewt", "cloaca_set1")
model_path = os.path.join(
    base,
    "crestedNewtProject",
    "machineLearningModels",
    "segmentTrain4",
    "weights",
    "last.pt",
)

model = YOLO(model_path)
dataset = NewtsKent(root, load_label=True)
dataset.df = dataset.df.iloc[:10]

os.chdir(root)

dataset.df[
    [
        "rulerOrigin",
        "pixelToCm",
        "progressionRatio",
        "rulerPoints",
        "rulerDirection",
    ]
] = dataset.df.apply(
    lambda x: RulerInference.infer_and_draw(x["path"], 768),
    result_type="expand",
    axis="columns",
)

dataset.df["path"].to_csv(
    "testExport.txt", sep="\n", index=False, header=False
)

filesToProcess = os.path.join(os.getcwd(), "testExport.txt")

# results = model.predict(filesToProcess, batch=10, stream=True)
results = model.predict(filesToProcess, batch=10)

pointValues = []
pathValues = []
for result in results:
    pathValues.append(os.path.relpath(result.path, root))
    pointValues.append(
        FindMeasureKeypoints.get_point_sequence(
            result, ["fake", "list", "of", "classes"]
        )
    )
df2 = pd.DataFrame({"lengthMeasurePoints": pointValues, "path": pathValues})

dataset.df = pd.merge(dataset.df, df2, how="inner", on="path")

dataset.df["length"] = dataset.df.apply(
    measureLength.measure_animal, axis="columns"
)
