import math
import numpy as np
def roi_pool(feature_map, rois, output_size):
    """
    Apply ROI Pooling to extract fixed-size features.
    """
    feature_map = np.asarray(feature_map)

    H, W = feature_map.shape
    output = np.zeros((len(rois), output_size, output_size))

    for idx, roi in enumerate(rois):
        x1, y1, x2, y2 = roi

        roi_h = y2 - y1
        roi_w = x2 - x1

        for i in range(output_size):
            for j in range(output_size):

                h_start = y1 + math.floor(i * roi_h / output_size)
                h_end = y1 + math.floor((i + 1) * roi_h / output_size)

                w_start = x1 + math.floor(j * roi_w / output_size)
                w_end = x1 + math.floor((j + 1) * roi_w / output_size)

                h_start = max(0, min(h_start, H))
                h_end = max(h_start + 1, min(h_end, H))

                w_start = max(0, min(w_start, W))
                w_end = max(w_start + 1, min(w_end, W))

                output[idx, i, j] = np.max(
                    feature_map[h_start:h_end, w_start:w_end]
                )

    return output.tolist()