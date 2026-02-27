# -*- coding: utf-8 -*-
"""
Created on Mon Feb 23 14:43:14 2026

@author: karl
"""

import RulerInference
import FindMeasureKeypoints
import measureLength
import numpy as np
import os
import pandas as pd
from wildlife_datasets.datasets import NewtsKent
from ultralytics import YOLO


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
