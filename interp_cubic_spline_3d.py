import numpy as np
from scipy import interpolate, ndimage

def interp_cubic_spline_3d(points, values, xi):
    """3D cubic spline interpolation.

    Zeroes are returned for interpolated values requested outside of the domain
    of the input data.
    """

    # Replicate the same expansion of xi as used in
    # RegularGridInterpolator, so the input xi can be quite flexible.
    xi = interpolate.interpnd._ndim_coords_from_arrays(xi, ndim=3)
    xi_shape = xi.shape
    xi = xi.reshape(-1, 3)

    # map_coordinates uses the indices of the input data as coordinates. We
    # have therefore to transform our desired output coordinates to this
    # artificial coordinate system too.
    params = {'kind': 'cubic',
              'bounds_error': False,
              'fill_value': 'extrapolate'}
    x = interpolate.interp1d(
            points[0], np.arange(len(points[0])), **params)(xi[:, 0])
    y = interpolate.interp1d(
            points[1], np.arange(len(points[1])), **params)(xi[:, 1])
    z = interpolate.interp1d(
            points[2], np.arange(len(points[2])), **params)(xi[:, 2])
    coords = np.vstack([x, y, z])

    # map_coordinates only works for real data; split it up if complex.
    if 'complex' in values.dtype.name:
        real = ndimage.map_coordinates(values.real, coords, order=3)
        imag = ndimage.map_coordinates(values.imag, coords, order=3)
        result = real + 1j*imag
    else:
        result = ndimage.map_coordinates(values, coords, order=3)

    return result.reshape(xi_shape[:-1])
