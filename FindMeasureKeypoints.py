# -*- coding: utf-8 -*-
"""
Created on Thu Nov 20 16:25:24 2025

@author: karl
"""

import numpy as np
import cv2 as cv


def get_point_sequence(result):
    cloacaFound = False
    headFound = False

    if 0 in result.boxes.cls and ((2 in result.boxes.cls) or (3 in result.boxes.cls)):
        # TODO: fix this
        # This currently assumes that the head is left nad cloaca right        
        for entry in result:
            mask_xy = entry.masks.xy[0]
            mom = cv.moments(mask_xy)
            if mom["m00"] != 0:
                x = mom["m10"] / mom["m00"]
                y = mom["m01"] / mom["m00"]
            else:
                x = 0
                y = 0
            if (entry.boxes.cls[0] == 0) and (not headFound):
                headCenterxy = np.array([x, y])
                headTipxyIndex = mask_xy.argmin(axis=0)
                headTipxy = mask_xy[headTipxyIndex]
                headFound = True
            elif (entry.boxes.cls[0] == 2 or entry.boxes.cls[0] == 3) and not cloacaFound:
                cloacaCenterxy = np.array([x, y])
                cloacaTipxyIndex = mask_xy.argmax(axis=0)
                cloacaTipxy = mask_xy[cloacaTipxyIndex]
                cloacaFound = True
        if cloacaFound and headFound:
            # TODO: Update this method. Right now, I find the axis that best aligns
            # with the animal (x or y) and take the min or the max along that axis.
            bestIndex = np.argmax(abs(headCenterxy - cloacaCenterxy))
            # result.show()
            return np.array([headTipxy[bestIndex], cloacaTipxy[bestIndex]])
        else:
            return None
