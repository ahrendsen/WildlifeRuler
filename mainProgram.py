# -*- coding: utf-8 -*-
"""
Created on Mon Feb 23 14:43:14 2026

@author: karl
"""

import RulerInference
import FindMeasureKeypoints
import measureLength
import os
from wildlife_datasets.datasets import NewtsKent
from ultralytics import YOLO


root = "/data/wildlife_datasets/newts_kent"
n_test = 10

model_path = os.path.join("weights", "newt-detector.pt")

model = YOLO(model_path)
dataset = NewtsKent(root, load_label=True)

dataset.set_absolute_paths()

metadata = dataset.metadata[:n_test].copy()
metadata[
    [
        "rulerOrigin",
        "pixelToCm",
        "progressionRatio",
        "rulerPoints",
        "rulerDirection",
    ]
] = metadata.apply(
    lambda x: RulerInference.infer_and_draw(x["path"], 768),
    result_type="expand",
    axis="columns",
)

results = model.predict(metadata["path"].tolist(), batch=10)
metadata["lengthMeasurePoints"] = [FindMeasureKeypoints.get_point_sequence(result) for result in results]
metadata["length"] = metadata.apply(measureLength.measure_animal, axis="columns")
metadata.to_csv("results.csv", index=False)
