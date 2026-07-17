def iou(box_a, box_b):
    """
    Compute Intersection over Union of two bounding boxes.
    """
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    inter_width = max(0, x2 - x1)
    inter_height = max(0, y2 - y1)
    intersection = inter_width * inter_height

    area1 = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area2 = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])

    union = area1 + area2 - intersection

    if union == 0:
        return 0.0

    return intersection / union