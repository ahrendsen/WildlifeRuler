import cv2
import networkx as nx
import numpy as np
import torch
from scipy.spatial import ConvexHull, distance_matrix
from scipy.ndimage import binary_erosion, label
from scipy.spatial.distance import cdist
from skimage.morphology import skeletonize
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



def filter_small_components(labeled, min_area):
    labels = np.unique(labeled)
    labels = labels[labels != 0]

    areas = {l: np.sum(labeled == l) for l in labels}
    keep = [l for l in labels if areas[l] >= min_area]

    out = np.zeros_like(labeled)
    for new_label, old_label in enumerate(keep, start=1):
        out[labeled == old_label] = new_label

    return out, len(keep)


# TODO: axis_1 is not nice
def fill_mask_axis_1(mask, tol_ratio=0.25, min_area=100):
    labeled, n = label(mask)
    labeled, n = filter_small_components(labeled, min_area)

    if n != 2:
        return mask

    mask1 = labeled == 1
    mask2 = labeled == 2
    idx1 = np.where(mask1)
    idx2 = np.where(mask2)
    if np.mean(idx1[1]) > np.mean(idx2[1]):
        mask1, mask2 = mask2, mask1
    
    col_min = []
    col_max = []
    for i in range(mask.shape[0]):
        cols1 = np.where(mask1[i])[0]
        cols2 = np.where(mask2[i])[0]
        col_min.append(cols1[-1] if len(cols1) > 0 else -np.inf)
        col_max.append(cols2[0] if len(cols2) > 0 else np.inf)

    border_diff = np.min(np.array(col_max) - np.array(col_min))
    
    if border_diff < 0:
        return mask
    
    filled = mask.copy()
    for i, (c_min, c_max) in enumerate(zip(col_min, col_max)):
        if c_max - c_min <= (1 + tol_ratio) * border_diff:
            filled[i, c_min:c_max+1] = True
    return filled


def get_centerline(mask: None | np.ndarray) -> None | np.ndarray:
    if mask is None:
        return None
    
    mask = mask.copy()
    mask = (mask > 0).astype(np.uint8)

    # Skeletonize
    skeleton = skeletonize(mask).astype(np.uint8)

    # Build graph from skeleton pixels
    G = nx.Graph()
    ys, xs = np.where(skeleton)
    for y, x in zip(ys, xs):
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0:
                    continue
                ny, nx_ = y + dy, x + dx
                if 0 <= ny < skeleton.shape[0] and 0 <= nx_ < skeleton.shape[1]:
                    if skeleton[ny, nx_]:
                        G.add_edge((y, x), (ny, nx_))

    # Find endpoints (degree == 1)
    endpoints = [n for n in G.nodes if G.degree[n] == 1]

    # Longest path between endpoints (head ↔ tail)
    max_path = []
    for i in range(len(endpoints)):
        for j in range(i + 1, len(endpoints)):
            try:
                path = nx.shortest_path(G, endpoints[i], endpoints[j])
                if len(path) > len(max_path):
                    max_path = path
            except nx.NetworkXNoPath:
                pass

    return np.array(max_path)


def longest_distance_one_mask(mask: None | np.ndarray) -> None | np.ndarray:
    if mask is None:
        return None
    
    pts = np.argwhere(mask)
    if len(pts) < 2:
        return None

    hull = pts[ConvexHull(pts).vertices]
    i, j = max_coords(distance_matrix(hull, hull))

    return np.array((hull[i], hull[j]))


def longest_distance_two_masks(mask1: None | np.ndarray, mask2: None | np.ndarray) -> None | np.ndarray:
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
