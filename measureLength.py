# -*- coding: utf-8 -*-
"""
Created on Wed Feb 25 13:27:21 2026

@author: karl
"""
import numpy as np
import copy
import os
from PIL import ImageDraw, Image


def animate_measurement():
    return None


def draw_annotations(df_entry, root):
    # ---- Annotate Image to demonstrate inference ----
    print(os.path.join(root, df_entry["path"]))
    image_pil = Image.open(os.path.join(root, df_entry["path"]))
    draw = ImageDraw.Draw(image_pil)
    rulerDirection = df_entry["rulerDirection"]
    pixelToCm = df_entry["pixelToCm"]
    rulerOrigin = df_entry["rulerOrigin"]
    perpendicular_direction = (
        -rulerDirection[1],
        rulerDirection[0],
    )  # shape: (2,)

    r = 15
    tickLength = 150
    for x, y in df_entry["rulerPoints"]:
        draw.ellipse((x - r, y - r, x + r, y + r), outline="red", width=3)
        draw.line(
            (
                (x, y),
                (
                    x + perpendicular_direction[0] * tickLength,
                    y + perpendicular_direction[1] * tickLength,
                ),
            ),
            width=5,
            fill="red",
        )

    x, y = rulerOrigin
    draw.ellipse((x - r, y - r, x + r, y + r), fill="green")
    draw.line(((x, y), (x, y + 50)), width=5, fill="red")

    for point in df_entry["lengthMeasurePoints"]:
        x, y = point
        draw.ellipse((x - r, y - r, x + r, y + r), fill="red")

    text = f"Pix/cm: {pixelToCm:.2f}"
    font_size = 100
    text_position = (10, 10)
    text_size = draw.textbbox(text_position, text, font_size=font_size)
    padding = 4
    rect_coords = (
        text_size[0] - padding,
        text_size[1] - padding,
        text_size[2] + padding,
        text_size[3] + padding,
    )
    draw.rectangle(rect_coords, fill="white")
    draw.text(text_position, text, fill="black", font_size=font_size)
    return image_pil


def measure_animal(df_entry):
    points = df_entry["lengthMeasurePoints"]
    rulerOrigin = df_entry["rulerOrigin"]
    cmConversion = df_entry["pixelToCm"]
    rulerDirection = df_entry["rulerDirection"]
    gpr = df_entry["progressionRatio"]  # The geometric progression ratio
    try:
        tip, tail = points[0], points[-1]
    except TypeError as e:
        print(f"There was a {type(e)}")
        return 0
    bodyVector = tip - tail
    headCloacaLine = (bodyVector) / np.linalg.norm(bodyVector)

    lengthIndex = np.argmax(abs(bodyVector))
    lengthMeasure = 0.0
    measurePoint = np.array([0.0, 0.0])
    measurePoint = copy.deepcopy(rulerOrigin)
    i = 0

    newDist = copy.deepcopy(cmConversion)
    # If measurement point is less than both the tip and tail.
    if (
        measurePoint[lengthIndex] <= tip[lengthIndex]
        and measurePoint[lengthIndex] <= tail[lengthIndex]
    ):
        # Increase measurePoint to be inside of tip-tail range
        while (measurePoint[lengthIndex] <= tip[lengthIndex]) and (
            measurePoint[lengthIndex] <= tail[lengthIndex]
        ):
            step = cmConversion * gpr ** (i)
            measurePoint += step * abs(headCloacaLine)
            newDist = step
            i += abs(np.dot(headCloacaLine, rulerDirection))
    elif (  # The measurement point is greater than both tip and tail
        measurePoint[lengthIndex] >= tip[lengthIndex]
        and measurePoint[lengthIndex] >= tail[lengthIndex]
    ):
        # Decrease measurePoint to be inside of tip-tail range
        while (measurePoint[lengthIndex] >= tip[lengthIndex]) and (
            measurePoint[lengthIndex] >= tail[lengthIndex]
        ):
            step = cmConversion * gpr ** (i)
            measurePoint += -step * abs(headCloacaLine)
            newDist = step
            # i += geoProDir  # Old way
            i += -abs(np.dot(headCloacaLine, rulerDirection))
    # print(f"measurePoint2: {measurePoint}")

    measurementStart = copy.deepcopy(measurePoint)
    i = 0
    # Now with measurepoint starting between tip and tail...
    # Measure the distance by incrementally increasing the
    # the measurement point until we're out of the measure range.
    while (measurePoint[lengthIndex] <= tail[lengthIndex]) or (
        measurePoint[lengthIndex] <= tip[lengthIndex]
    ):
        step = 0.01 * newDist * gpr ** (i * 0.01)
        lengthMeasure += 0.01
        measurePoint += step * abs(headCloacaLine)
        # i += geoProDir
        i += abs(np.dot(headCloacaLine, rulerDirection))
    # Measure the distance by incrementally DECREASING the
    # the measurement point until we're out of the measure range.
    i = 0
    measurePoint = copy.deepcopy(measurementStart)
    # print(f"One direction Length Measure: {lengthMeasure}")
    while (measurePoint[lengthIndex] >= tip[lengthIndex]) or (
        measurePoint[lengthIndex] >= tail[lengthIndex]
    ):
        step = 0.01 * newDist * gpr ** (i * 0.01)
        lengthMeasure += 0.01
        measurePoint += -step * abs(headCloacaLine)
        # i += -geoProDir
        i += -abs(np.dot(headCloacaLine, rulerDirection))
    return lengthMeasure
