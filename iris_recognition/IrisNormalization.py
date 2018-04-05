import numpy as np


def run(img, pupil, outer):
    in_col, in_row, in_r = pupil
    out_col, out_row, out_r = outer
    r = int(round(out_r - np.hypot(in_col-out_col, in_row-out_row)))
    if r < in_r + 15:
        r = 80
    maxh, maxw = img.shape
    r2 = min(in_col, in_row, maxh-in_row, maxw-in_col)
    r = min(r, r2)
    max_r = r - in_r
    M = 64
    N = 512
    normalized = []
    for y in range(M):
        row_pix = []
        for x in range(N):
            theta = float(2*np.pi*x/N)
            hypotenuse = float(max_r*y/M) + in_r
            col = int(round((np.cos(theta) * hypotenuse) + in_col))
            row = int(round((np.sin(theta) * hypotenuse) + in_row))
            row_pix.append(img[row,col])
        normalized.append(row_pix)

    return np.array(normalized)
