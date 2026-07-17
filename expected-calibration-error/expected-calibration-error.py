def expected_calibration_error(y_true, y_pred, n_bins):
    """
    Compute Expected Calibration Error.
    """

    bin_width = 1 / n_bins
    bins = []
    for i in range(n_bins):
        bins.append([bin_width*i, bin_width*(i+1)])

    bm = {}

    for i in range(len(y_pred)):
        for bin in bins:
            if bin[0] <= y_pred[i] < bin[1]:
                bm[bin[0]] = bm.get(bin[0], [])
                bm[bin[0]].append(i)

    ECE = 0
    for bin, samples in bm.items():
        sum_y_true = 0
        sum_y_pred = 0
        for i in samples:
            sum_y_true+= y_true[i]
            sum_y_pred+= y_pred[i]
        acc = sum_y_true / len(samples)
        con = sum_y_pred / len(samples)

        ECE += len(samples)*abs(acc - con) / len(y_true)

    return ECE

    