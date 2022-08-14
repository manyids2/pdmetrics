from typing import Dict, Tuple
from rich import print
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw
from distinctipy import distinctipy


def get_colors_from_labels(
    labels: Dict[int, str],
    background_idx: int = 0,
) -> Tuple[Dict[int, Tuple[int, int, int]], np.ndarray]:
    """Generate distinct colors as 3-tuple of ints in range [0, 255]."""
    _colors = distinctipy.get_colors(len(labels), rng=0)
    colors = {
        k: tuple(int(c * 255) for c in _colors[idx]) for idx, k in enumerate(labels)
    }

    # Set background to zero
    colors[background_idx] = (0, 0, 0)

    legend = np.zeros([len(labels) * 100, 100, 3], dtype=np.uint8)
    for k, color in colors.items():
        legend[k * 100 : (k + 1) * 100, :] = color

    # Draw the text on top using PIL
    _legend = ImageDraw.Draw(Image.fromarray(legend))
    for k, label in labels.items():
        _legend.text((k * 100, 0), label)
    legend = np.array(_legend)

    return colors, legend


# Get 50 labels and colors
LABELS = {
    0: "background",
}
LABELS.update({k: str(k) for k in range(50)})
COLORS, LEGEND = get_colors_from_labels(LABELS)


def to_numpy(x):
    return x.detach().cpu().numpy()


def blend(image, mask, alpha):
    return Image.blend(Image.fromarray(image), Image.fromarray(mask), alpha=alpha)


def mask_to_colors(mask, colors):
    idxs = list(set(mask.flatten().tolist()))
    render = np.zeros([*mask.shape, 3], dtype=np.uint8)
    for class_idx in idxs:
        class_idx = int(class_idx)
        render[mask == class_idx] = colors[class_idx]
    return render


def show_save(pil_image, show=True, save=None):
    if show:
        pil_image.show()

    if save is not None:
        pil_image.save(str(save))


def to_pil_image(image, itype="ndarray", isfloat=False):
    """
    Convert ndarray or torch tensor to PIL image.
    """

    # First convert to numpy
    if (itype == "ndarray") | (itype == "numpy"):
        _image = image
    elif (itype == "tensor-hwc") | (itype == "torch-hwc"):
        _image = to_numpy(image)
    elif (itype == "tensor-chw") | (itype == "torch-chw"):
        _image = to_numpy(image.permute(1, 2, 0))
    else:
        raise KeyError(f"{itype} is not a valid itype.")

    # Check if it is floating point
    if isfloat:
        _image = _image * 255

    # Make sure dtype is uint8
    _image = _image.astype(np.uint8)

    pil_image = Image.fromarray(_image)
    return pil_image


def draw_image_boxes(
    pil_image: Image.Image,
    boxes: np.ndarray,
    colors: Dict[int, Tuple],
    linewidth=2,
    filled=False,
) -> Image.Image:
    """
    Draw boxes {y1, x1, y2, x2, label} on image ( PIL Image ).

    Params:
    pil_image: PIL.Image.Image
    boxes: np.ndarray, {y1, x1, y2, x2, label}, [n, 5], float32
    """
    draw = ImageDraw.Draw(pil_image)
    for box in boxes:
        x1, y1, x2, y2, label = box
        fill = colors[int(label)] if filled else None
        draw.rectangle(
            (x1, y1, x2, y2), fill=fill, outline=colors[int(label)], width=linewidth
        )
    return pil_image


def draw_image_circles(
    pil_image: Image.Image,
    circles: np.ndarray,
    colors: Dict[int, Tuple],
    linewidth=2,
    filled=False,
) -> Image.Image:
    """
    Draw circles {y, x, label} on image ( PIL Image ).

    Params:
    pil_image: PIL.Image.Image
    circles: np.ndarray, {y, x, label}, [n, 3], float32
    """
    draw = ImageDraw.Draw(pil_image)
    for circle in circles:
        x1, y1, x2, y2, label = circle
        # (y, x, l), r = circle, radius
        fill = colors[int(label)] if filled else None
        draw.ellipse(
            (x1, y1, x2, y2),
            fill=fill,
            outline=colors[int(label)],
            width=linewidth,
        )
    return pil_image


def draw_image_mask(
    pil_image: Image.Image,
    mask: np.ndarray,
    colors: Dict[int, Tuple],
    alpha: float = 0.3,
) -> Image.Image:
    """
    Draw mask {integer per pixel} on image ( PIL Image ).

    Params:
    pil_image: PIL.Image.Image
    mask: np.ndarray, {integer per pixel}, [h, w], uint8
    """
    render = mask_to_colors(mask, colors)
    overlay = Image.blend(pil_image, Image.fromarray(render), alpha=alpha)
    return overlay


def draw_image_polygon(
    pil_image: Image.Image,
    polygons: Dict[int, np.ndarray],
    colors: Dict[int, Tuple],
    fills=None,
    linewidth=1,
) -> Image.Image:
    """
    Draw polygon {list of [x, y]} on image ( PIL Image ).

    Params:
    pil_image: PIL.Image.Image
    polygon: np.ndarray, [n, 2]
    """
    draw = ImageDraw.Draw(pil_image)
    for label, poly in polygons.items():
        if fills is not None:
            fill = fills[int(label)]
        else:
            fill = None
        draw.polygon(
            [tuple(p) for p in poly.tolist()],
            fill=fill,
            outline=colors[int(label)],
            width=linewidth,
        )
    return pil_image


SCHEMA: Dict[str, str] = {
    "image_file": "path-image",
    "mask_file": "path-image",
    "overlay_file": "path-image",
    "boxes_file": "path-npz",
    "slide_file": "path-wsi",
    "ratings": "int-ratings",
    "comments": "str-comments",
    "source": "str-source",
    "split": "str-split",
}


class pdViz:
    def __init__(self, df: pd.DataFrame, schema: Dict[str, str]):
        self.df = df
        print(schema)
