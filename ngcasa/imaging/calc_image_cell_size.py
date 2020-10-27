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
"""
this module will be included in the api
"""

def calc_image_cell_size(vis_dataset, global_dataset,pixels_per_beam=7):
    """
    Calculates the image and and cell size needed for imaging a vis_dataset.
    It uses the perfectly-illuminated circular aperture approximation to determine the field of view
    and pixels_per_beam for the cell size.

    Parameters
    ----------
    vis_dataset : xarray.core.dataset.Dataset
        Input visibility dataset.
    global_dataset : xarray.core.dataset.Dataset
        Input global dataset (needed for antenna diameter).
    Returns
    -------
    imsize : list of ints
        Number of pixels for each spatial dimension.
    cell : list of ints, units = arcseconds
        Cell size.
    """
    import xarray
    import numpy as np
    import dask.array  as da
    rad_to_arc = (3600 * 180) / np.pi  # Radians to arcseconds
    c = 299792458

    f_min = da.nanmin(vis_dataset.chan)
    f_max = da.nanmax(vis_dataset.chan)
    D_min = np.nanmin(global_dataset.ANT_DISH_DIAMETER)
    #D_min = min_dish_diameter

    # Calculate cell size using pixels_per_beam
    cell = rad_to_arc * np.array(
        [c / (da.nanmax(vis_dataset.UVW[:, :, 0].data) * f_max), c / (da.nanmax(vis_dataset.UVW[:, :, 1].data) * f_max)]) / pixels_per_beam

    # If cell sizes are within 20% of each other use the smaller cell size for both.
    if (cell[0] / cell[1] < 1.2) and (cell[1] / cell[0] < 1.2):
        cell[:] = np.min(cell)

    # Calculate imsize using the perfectly-illuminated circular aperture approximation
    FWHM_max = np.array((rad_to_arc * (1.02 * c / (D_min * f_min))))
    imsize = FWHM_max / cell

    # Find an image size that is (2^n)*10 when muliplied with the gridding padding and n is an integer
    padding = 1.2

    if imsize[0] < 1:
        imsize[0] = 1

    if imsize[1] < 1:
        imsize[1] = 1

    n_power = np.ceil(np.log2(imsize / 10))
    imsize = np.ceil(((2 ** n_power) * 10) / padding)

    return cell, imsize
