import numpy as np
def histogram_equalize(image):
    """
    Apply histogram equalization to enhance image contrast.
    """
    image = np.asarray(image, dtype=np.uint8)
    hist = np.bincount(image.ravel(), minlength=256)

    # CDF
    cdf = np.cumsum(hist)

    cdf_min = cdf[np.nonzero(cdf)[0][0]]

    total_pixels = image.size
    if total_pixels == cdf_min:
        return np.zeros_like(image).tolist()

    # Lookup table
    lut = np.round(
        (cdf - cdf_min) / (total_pixels - cdf_min) * 255
    )

    lut = np.clip(lut, 0, 255).astype(np.uint8)

    # Apply mapping
    output = lut[image]

    return output.tolist()