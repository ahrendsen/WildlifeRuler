import numpy as np
import torch
from scipy.spatial import ConvexHull, distance_matrix
from scipy.ndimage import binary_erosion
from scipy.spatial.distance import cdist
from ultralytics.engine.results import Results

def best_mask_for_class(result: Results, class_ids: list[int]) -> None | np.ndarray:
    if result.masks is None or result.boxes is None:
        return None

    conf = result.boxes.conf
    cls = torch.as_tensor(result.boxes.cls)

    class_ids_tensor = torch.as_tensor(class_ids, device=cls.device)
    valid = torch.isin(cls, class_ids_tensor)
    idx = valid.nonzero(as_tuple=True)[0]
    if len(idx) == 0:
        return None

    best = idx[conf[idx].argmax()]
    mask = result.masks.data[best]
    return to_binary_mask(mask)


def boundary(mask: np.ndarray) -> np.ndarray:
    eroded = binary_erosion(mask)
    return mask & ~eroded


def longest_distance_one_mask(mask: np.ndarray) -> None | np.ndarray:
    if mask is None:
        return None
    
    pts = np.argwhere(mask)
    if len(pts) < 2:
        return None

    hull = pts[ConvexHull(pts).vertices]
    i, j = max_coords(distance_matrix(hull, hull))

    return np.array((hull[i], hull[j]))


def longest_distance_two_masks(mask1: np.ndarray, mask2: np.ndarray) -> None | np.ndarray:
    if mask1 is None or mask2 is None:
        return None
    
    b1 = np.argwhere(boundary(mask1))
    b2 = np.argwhere(boundary(mask2))
    
    if len(b1) == 0 or len(b2) == 0:
        return None

    i, j = max_coords(cdist(b1, b2))

    return np.array((b1[i], b2[j]))


def max_coords(A: np.ndarray):
    return np.unravel_index(np.argmax(A), A.shape)


def to_binary_mask(mask) -> np.ndarray:
    mask_bool = mask.data.cpu().detach().bool()
    if mask_bool.ndim == 2:
        return mask_bool.numpy()
    elif mask_bool.ndim == 3:
        assert mask_bool.shape[0] == 1
        return mask_bool[0].numpy()
    raise ValueError("Wrong input shape")
