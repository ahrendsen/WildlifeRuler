import os
import numpy as np
from wildlife_datasets import datasets
from ..measure.centerline import best_mask_for_class, get_centerline, fill_mask_axis_1, longest_distance_one_mask, longest_distance_two_masks


# TODO: move it
import cv2
def resize_mask(mask, new_shape):
    return cv2.resize(
        mask.astype(np.uint8),
        (new_shape[1], new_shape[0]),
        interpolation=cv2.INTER_NEAREST
    ).astype(bool)


class Dataset:
    def __init__(self, root, **kwargs):
        self.dataset = datasets.Dataset_Folder(root, **kwargs)
        self.dataset.set_absolute_paths()

    def __getitem__(self, i):
        return self.dataset[i]


class Dataset_SAM3(Dataset):
    def __init__(self, root, prompt, model, processor, min_area=100, **kwargs):
        self.prompt = prompt
        self.model = model        
        self.processor = processor
        self.min_area = min_area
        super().__init__(root, **kwargs)

    def _extract_centerline(self, mask) -> None | np.ndarray:
        raise NotImplementedError("Must be implemented by subclasses")

    def _get_masks(self, i: int) -> list[np.ndarray]:
        image = self[i]
        inference_state = self.processor.set_image(image)
        self.processor.reset_all_prompts(inference_state)
        inference_state = self.processor.set_text_prompt(state=inference_state, prompt=self.prompt)

        masks = []
        for m in inference_state["masks"]:
            m = m.cpu().numpy().astype(bool)
            if m.ndim == 3 and m.shape[0] == 1:
                m = m[0]
            if m.sum() > self.min_area:
                masks.append(m)
        return masks

    def extract_centerline(self, i: int) -> None | np.ndarray:
        masks = self._get_masks(i)
        if len(masks) == 0:
            return None
        elif len(masks) > 1:
            print(f"Image {i} has too many masks. Taking only the first one.")
        return self._extract_centerline(masks[0])


class Fish(Dataset_SAM3):
    def _extract_centerline(self, mask) -> None | np.ndarray:
        return longest_distance_one_mask(mask)


class Newts(Dataset):
    def __init__(self, root, segmentation_model):
        self.dataset = datasets.NewtsKent(root)
        self.dataset.set_absolute_paths()
        self.segmentation_model = segmentation_model
    
    def extract_centerline(self, i: int) -> None | np.ndarray:
        path = self.dataset.metadata["path"].iloc[i]
        result = self.segmentation_model.predict(path)[0]

        mask_head = best_mask_for_class(result, [0])
        mask_cloaca = best_mask_for_class(result, [2, 3])
        if mask_head is None or mask_cloaca is None:
            return None

        mask_head = resize_mask(mask_head, result.orig_shape)
        mask_cloaca = resize_mask(mask_cloaca, result.orig_shape)

        return longest_distance_two_masks(mask_head, mask_cloaca)


class SeaHorses(Dataset_SAM3):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        mask = self.dataset.metadata["path"].apply(lambda x: os.path.splitext(x)[0].endswith("A"))
        self.dataset = self.dataset.get_subset(mask)

    def _extract_centerline(self, mask) -> None | np.ndarray:
        mask = fill_mask_axis_1(mask)
        return get_centerline(mask)


class Snakes(Dataset_SAM3):
    def _extract_centerline(self, mask) -> None | np.ndarray:
        return get_centerline(mask)
