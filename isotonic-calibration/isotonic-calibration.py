from bisect import bisect_right
def calibrate_isotonic(cal_labels, cal_probs, new_probs):
    """
    Apply isotonic regression calibration.
    """
    pairs = sorted(zip(cal_probs, cal_labels))
    probs = [p for p, _ in pairs]
    labels = [y for _, y in pairs]


    blocks = []

    for i,y in enumerate(labels):
        blocks.append(
            {
                "start":i,
                "end":i,
                "sum":y,
                "count":1,
                "mean":float(y)
            }
        )

        while len(blocks) >= 2 and blocks[-2]["mean"] > blocks[-1]["mean"]:
            b2 = blocks.pop()
            b1 = blocks.pop()

            total_sum = b1['sum'] + b2['sum']
            total_count = b1['count'] + b2['count']

            blocks.append({
                "start": b1["start"],
                "end": b2["end"],
                "sum": total_sum,
                "count":total_count,
                "mean": total_sum/total_count
            })

    calibrated = [0.0]*len(labels)

    for block in blocks:
        for i in range(block['start'], block['end']+1):
            calibrated[i] = block['mean']

    result= []

    for q in new_probs:

        if q < probs[0]:
            result.append(calibrated[0])
            continue

        if q >= probs[-1]:
            result.append(calibrated[-1])
            continue

        idx = bisect_right(probs, q)

        p1, p2 = probs[idx - 1], probs[idx]
        c1, c2 = calibrated[idx-1], calibrated[idx]

        if p1 == p2:
            result.append(c2)
        else:
            t = (q - p1) / (p2 - p1)
            result.append(c1 + t*(c2 - c1))
    return result