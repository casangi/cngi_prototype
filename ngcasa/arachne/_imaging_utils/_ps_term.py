#   Copyright 2019 AUI, Inc. Washington DC, USA
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
import numpy as np
from numba import jit
import math


def _coordinates(npixel: int):
    """ 1D array which spans [-.5,.5[ with 0 at position npixel/2
    """
    return (np.arange(npixel) - npixel // 2) / npixel


def _coordinates2(npixel: int):
    """Two dimensional grids of coordinates spanning -1 to 1 in each dimension
    1. a step size of 2/npixel and
    2. (0,0) at pixel (floor(n/2),floor(n/2))
    """
    return (np.mgrid[0:npixel, 0:npixel] - npixel // 2) / npixel


def _create_prolate_spheroidal_kernel(oversampling, support, n_uv):
    """
    Create PSWF to serve as gridding kernel

    Parameters
    ----------
    oversampling : int 
        oversampling//2 is the index of the zero value of the oversampling value
    support : int
        support//2 is the index of the zero value of the support values
    n_uv: int array
        (2)
        number of pixels in u,v space

    Returns
    -------
    kernel : numpy.ndarray
    kernel_image : numpy.ndarray

    """
    # support//2 is the index of the zero value of the support values
    # oversampling//2 is the index of the zero value of the oversampling value
    support_center = support // 2
    oversampling_center = oversampling // 2

    support_values = (np.arange(support) - support_center)
    if (oversampling % 2) == 0:
        oversampling_values = ((np.arange(oversampling + 1) - oversampling_center) / oversampling)[:, None]
        kernel_points_1D = (np.broadcast_to(support_values, (oversampling + 1, support)) + oversampling_values)
    else:
        oversampling_values = ((np.arange(oversampling) - oversampling_center) / oversampling)[:, None]
        kernel_points_1D = (np.broadcast_to(support_values, (oversampling, support)) + oversampling_values)

    kernel_points_1D = kernel_points_1D / support_center

    _, kernel_1D = _prolate_spheroidal_function(kernel_points_1D)
    # kernel_1D /= np.sum(np.real(kernel_1D[oversampling_center,:]))

    if (oversampling % 2) == 0:
        kernel = np.zeros((oversampling + 1, oversampling + 1, support, support),
                          dtype=np.double)  # dtype=np.complex128
    else:
        kernel = np.zeros((oversampling, oversampling, support, support), dtype=np.double)

    for x in range(oversampling):
        for y in range(oversampling):
            kernel[x, y, :, :] = np.outer(kernel_1D[x, :], kernel_1D[y, :])

    # norm = np.sum(np.real(kernel))
    # kernel /= norm

    # Gridding correction function (applied after dirty image is created)
    kernel_image_points_1D_u = np.abs(2.0 * _coordinates(n_uv[0]))
    kernel_image_1D_u = _prolate_spheroidal_function(kernel_image_points_1D_u)[0]

    kernel_image_points_1D_v = np.abs(2.0 * _coordinates(n_uv[1]))
    kernel_image_1D_v = _prolate_spheroidal_function(kernel_image_points_1D_v)[0]

    kernel_image = np.outer(kernel_image_1D_u, kernel_image_1D_v)

    # kernel_image[kernel_image > 0.0] = kernel_image.max() / kernel_image[kernel_image > 0.0]

    # kernel_image =  kernel_image/kernel_image.max()
    return kernel, kernel_image


def _prolate_spheroidal_function(u):
    """
    Calculate PSWF using an old SDE routine re-written in Python

    Find Spheroidal function with M = 6, alpha = 1 using the rational
    approximations discussed by Fred Schwab in 'Indirect Imaging'.

    This routine was checked against Fred's SPHFN routine, and agreed
    to about the 7th significant digit.

    The griddata function is (1-NU**2)*GRDSF(NU) where NU is the distance
    to the edge. The grid correction function is just 1/GRDSF(NU) where NU
    is now the distance to the edge of the image.
    """
    p = np.array([[8.203343e-2, -3.644705e-1, 6.278660e-1, -5.335581e-1, 2.312756e-1],
                  [4.028559e-3, -3.697768e-2, 1.021332e-1, -1.201436e-1, 6.412774e-2]])
    q = np.array([[1.0000000e0, 8.212018e-1, 2.078043e-1], [1.0000000e0, 9.599102e-1, 2.918724e-1]])

    _, n_p = p.shape
    _, n_q = q.shape

    u = np.abs(u)
    uend = np.zeros(u.shape, dtype=np.float64)
    part = np.zeros(u.shape, dtype=np.int64)

    part[(u >= 0.0) & (u < 0.75)] = 0
    part[(u >= 0.75) & (u <= 1.0)] = 1
    uend[(u >= 0.0) & (u < 0.75)] = 0.75
    uend[(u >= 0.75) & (u <= 1.0)] = 1.0

    delusq = u ** 2 - uend ** 2

    top = p[part, 0]
    for k in range(1, n_p):  # small constant size loop
        top += p[part, k] * np.power(delusq, k)

    bot = q[part, 0]
    for k in range(1, n_q):  # small constant size loop
        bot += q[part, k] * np.power(delusq, k)

    grdsf = np.zeros(u.shape, dtype=np.float64)
    ok = (bot > 0.0)
    grdsf[ok] = top[ok] / bot[ok]
    ok = np.abs(u > 1.0)
    grdsf[ok] = 0.0

    # Return the correcting image and the gridding kernel value
    return grdsf, (1 - u ** 2) * grdsf


def _create_prolate_spheroidal_kernel_1D(oversampling, support):
    support_center = support // 2
    oversampling_center = oversampling // 2
    u = np.arange(oversampling * (support_center)) / (support_center * oversampling)

    long_half_kernel_1D = np.zeros(oversampling * (support_center + 1))
    _, long_half_kernel_1D[0:oversampling * (support_center)] = _prolate_spheroidal_function(u)
    return long_half_kernel_1D

def _create_prolate_spheroidal_kernel_2D(oversampling, support):
    """
    Prolate spheroidal gridding term
    """

    conv_size = (support + 1)*oversampling
    half_support = support//2
    print(conv_size)
    
    oversampled_coordinates_x = np.abs((np.arange(conv_size[0]) - conv_size[0]//2)/(oversampling[0]*half_support[0]))
    oversampled_coordinates_y = np.abs((np.arange(conv_size[1]) - conv_size[1]//2)/(oversampling[1]*half_support[1]))
    
    oversampled_coordinates_x[oversampled_coordinates_x >= 1] = 1
    oversampled_coordinates_y[oversampled_coordinates_y >= 1] = 1
    
    kernel_image_1D_x = _prolate_spheroidal_function(oversampled_coordinates_x)[1]
    kernel_image_1D_y = _prolate_spheroidal_function(oversampled_coordinates_y)[1]
    
    kernel_image = np.outer(kernel_image_1D_x, kernel_image_1D_y)
    
    return kernel_image
    

def _create_prolate_spheroidal_image_2D(n_xy):
    """
    Correcting image for prolate spheroidal term
    """
    
    kernel_image_points_1D_x = np.abs(2.0 * _coordinates(n_xy[0]))
    kernel_image_1D_x = _prolate_spheroidal_function(kernel_image_points_1D_x)[0]

    kernel_image_points_1D_y = np.abs(2.0 * _coordinates(n_xy[1]))
    kernel_image_1D_y = _prolate_spheroidal_function(kernel_image_points_1D_y)[0]

    kernel_image = np.outer(kernel_image_1D_x, kernel_image_1D_y)

    return kernel_image
