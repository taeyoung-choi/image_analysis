import numpy as np


def run(img, pupil, outer):
    """
    Irises from different people may be captured in different size and, even for irises from the same eye,
    the size may change due to illumination variations and other factors.

    This step normalizes the iris area to a rectangular image of size 64 by 512

    :param img: gray-scale eye image
    :param pupil: pupil center and radius
    :param outer: outer boundary center and radius
    :return: normalized iris image
    """
    in_col, in_row, in_r = pupil
    out_col, out_row, out_r = outer

    # Dectected circles for the inner boundary and the outer boundary of an iris is usually not concentric
    # Calculate the closest point from the pupil center to the outer boundary
    r = int(round(out_r - np.hypot(in_col-out_col, in_row-out_row)))

    # there are two error cases:
    # 1. Outer boundary is wrongly defined.
    if r < in_r + 15:
        # manually assign the max distance from the pupil center
        r = 80

    # 2. Outer boundary does not fit in the image
    # Calculate the closest distance from the center to all image boundaries.
    maxh, maxw = img.shape

    r2 = min(in_col, in_row, maxh-in_row, maxw-in_col)
    # Define the region that we want to normalize
    r = min(r, r2)
    max_r = r - in_r
    # Normalize the iris area to a rectangular image of size 64 by 512
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
            row_pix.append(img[row, col])
        normalized.append(row_pix)

    return np.array(normalized)
