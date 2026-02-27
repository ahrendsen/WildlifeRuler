# -*- coding: utf-8 -*-
"""
Created on Thu Nov 20 16:25:24 2025

@author: karl
"""
import os
import numpy as np
import pandas as pd
import cv2 as cv
import matplotlib.pyplot as plt
from PIL import Image
import re


def get_point_sequence(result, list_of_classes):
    cloacaFound = False
    headFound = False
    class0 = list_of_classes[0]
    class1 = list_of_classes[1]
    class2 = list_of_classes[2]
    try:
        if len(list_of_classes) < 2:
            raise Exception("List not long enough.")
    except:
        print("At least two class names must be provided.")
    result.names

    if (0 in result.boxes.cls) and (
        (2 in result.boxes.cls) or (3 in result.boxes.cls)
    ):
        for entry in result:
            mom = cv.moments(entry.masks.xy[0])
            if mom["m00"] != 0:
                x = mom["m10"] / mom["m00"]
                y = mom["m01"] / mom["m00"]
            else:
                x = 0
                y = 0
            if (entry.boxes.cls[0] == 0) and (not headFound):
                headCenterxy = np.array([x, y])
                headTipxyIndex = np.array(
                    # This currently assumes that the head will be on
                    # the top or the left of the image.
                    [
                        np.argmin(entry.masks.xy[0][:, 0]),
                        np.argmin(entry.masks.xy[0][:, 1]),
                    ]
                )
                headTipxy = np.array(
                    [
                        entry.masks.xy[0][headTipxyIndex[0]],
                        entry.masks.xy[0][headTipxyIndex[1]],
                    ]
                )
                headCert = entry.boxes.conf[0].numpy()
                headFound = True
            elif (entry.boxes.cls[0] == 2 or entry.boxes.cls[0] == 3) and (
                not cloacaFound
            ):
                # This currently assumes that the cloaca will be on
                # the bottom or the right of the image.
                cloacaCenterxy = np.array([x, y])
                cloacaTipxyIndex = np.array(
                    [
                        np.argmax(entry.masks.xy[0][:, 0]),
                        np.argmax(entry.masks.xy[0][:, 1]),
                    ]
                )
                cloacaTipxy = np.array(
                    [
                        entry.masks.xy[0][cloacaTipxyIndex[0]],
                        entry.masks.xy[0][cloacaTipxyIndex[1]],
                    ]
                )
                cloacaCert = entry.boxes.conf[0].numpy()
                cloacaFound = True
        if cloacaFound and headFound:
            # TODO: Update this method. Right now, I find the axis that best aligns
            # with the animal (x or y) and take the min or the max along that axis.
            bestIndex = np.argmax(abs(headCenterxy - cloacaCenterxy))
            # result.show()
            return np.array([headTipxy[bestIndex], cloacaTipxy[bestIndex]])
        else:
            return np.array([[0, 0], [0, 0]])


if __name__ == "__main__":
    base = "C:\\home\\programming\\neuralNewtwork"
    root = os.path.join(base, "datasets", "crestedNewt", "cloaca_set1")
