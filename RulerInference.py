import os
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw


# Download the private model.onnx file
model_path = "./weights/model.onnx"

# ---- Load ONNX Model ----
ort_session = ort.InferenceSession(
    model_path, providers=["CPUExecutionProvider"]
)


# ---- Utility Function ----
def outward_cumsum(initial_point, line_direction, directions, n):
    left_directions = directions[n < 0][::-1]
    right_directions = directions[n >= 0]

    left_increments = -np.expand_dims(left_directions, axis=1) * line_direction
    right_increments = (
        np.expand_dims(right_directions, axis=1) * line_direction
    )

    zero = np.zeros((1, 2), dtype=initial_point.dtype)
    left_cumulative = np.cumsum(np.vstack([zero, left_increments]), axis=0)
    right_cumulative = np.cumsum(np.vstack([zero, right_increments]), axis=0)

    left_points = initial_point + left_cumulative
    right_points = initial_point + right_cumulative

    extended_points = np.vstack([left_points[::-1], right_points[1:]])
    return extended_points


def rescale_to_original(pixel_array, left, top, scale):
    return (pixel_array - np.array([left, top])) / scale


# ---- Main Inference and Drawing ----
def infer_and_draw(image_path, infer_image_size=768):
    image_pil = Image.open(image_path)
    image = np.array(image_pil).astype(np.float32) / 255.0
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    elif image.shape[2] == 4:
        image = image[:, :, :3]

    # Image preprocessing for inference.
    resized = np.zeros(
        (infer_image_size, infer_image_size, 3), dtype=np.float32
    )
    h, w = image.shape[:2]
    scale = min(infer_image_size / w, infer_image_size / h)
    new_w, new_h = int(w * scale), int(h * scale)
    image_resized = (
        np.array(
            Image.fromarray((image * 255).astype(np.uint8)).resize(
                (new_w, new_h)
            )
        ).astype(np.float32)
        / 255.0
    )
    top = (infer_image_size - new_h) // 2
    left = (infer_image_size - new_w) // 2
    resized[top : top + new_h, left : left + new_w] = image_resized

    input_tensor = np.transpose(resized, (2, 0, 1))[np.newaxis, ...].copy()
    image_tensor = input_tensor
    # End Image preprocessing

    # Get Model Results
    ort_inputs = {"input": image_tensor}
    ort_outs = ort_session.run(None, ort_inputs)

    left_point_2d_reconstructed = ort_outs[0][0]  # shape: (2,)
    dist = ort_outs[1][0][0]  # scalar
    ratio = ort_outs[2][0][0]  # scalar
    direction = ort_outs[3][0]  # shape: (2,)
    perpendicular_direction = (
        -ort_outs[3][0][1],
        ort_outs[3][0][0],
    )  # shape: (2,)
    points_info = ort_outs[4][0]

    min_x, min_y, max_x, max_y = points_info[1:].tolist()
    num_points = int(points_info[0])

    n = np.arange(-num_points, num_points + 1)
    directions = (ratio**n) * dist

    extended_points = outward_cumsum(
        left_point_2d_reconstructed, direction, directions, n
    )

    within_bounds = (
        (extended_points[:, 0] >= min_x)
        & (extended_points[:, 0] <= max_x)
        & (extended_points[:, 1] >= min_y)
        & (extended_points[:, 1] <= max_y)
    )
    best_generated_points = extended_points[within_bounds]

    # Convert points to original image pixels
    best_generated_points = rescale_to_original(
        best_generated_points, left, top, scale
    )

    if len(best_generated_points) > 1:
        diffs = np.linalg.norm(
            best_generated_points[:-1] - best_generated_points[1:], axis=1
        )
        pred_pix_cm = np.nanmedian(diffs)
    else:
        pred_pix_cm = 0.0

    return (
        rescale_to_original(left_point_2d_reconstructed, left, top, scale)[0],
        dist / scale,
        ratio,
        best_generated_points,
        direction,
    )


if __name__ == "__main__":
    imagePath = os.path.join(
        "C:\\\\",
        "home",
        "programming",
        "neuralNewtwork",
        "datasets",
        "crestedNewt",
        "cloaca_set1",
        "val",
        "male",
        "2024",
        "IMG_5299_M2138.JPG",
    )

    origin_point, conversion, ratio, points, direction = infer_and_draw(
        imagePath, 768
    )
