def nms(boxes, scores, iou_threshold):
    """
    Apply Non-Maximum Suppression.
    """
    order = sorted(range(len(scores)),
                  key = lambda i: scores[i],
                  reverse=True)

    keep = []

    while order:
        i = order.pop(0)

        keep.append(i)

        remaining= []
        for j in order:

            x1 = max(boxes[i][0], boxes[j][0])
            y1 = max(boxes[i][1], boxes[j][1])
            x2 = min(boxes[i][2], boxes[j][2])
            y2 = min(boxes[i][3], boxes[j][3])

            inter = max(0,x2-x1)*max(0, y2-y1)

            area1 = (boxes[i][2] - boxes[i][0])*(boxes[i][3] - boxes[i][1])
            area2 = (boxes[j][2] - boxes[j][0])*(boxes[j][3] - boxes[j][1])

            union = area1 + area2 - inter
            if union == 0:
                iou = 0
            else:
                iou = inter / union

            if iou < iou_threshold:
                remaining.append(j)
        order = remaining
    return keep
                
        