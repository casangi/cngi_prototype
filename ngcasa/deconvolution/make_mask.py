#  CASA Next Generation Infrastructure
#  Copyright (C) 2021 AUI, Inc. Washington DC, USA
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
this module will be included in the api
"""

def make_mask(img_dataset,mask_parms,storage_parms):
    """
    .. todo::
        This function is not yet implemented
    
    Make a region to identify a mask for use in deconvolution.
    
    One or more of the following options are allowed
    
    - Supply a mask in the form of a cngi.image.region
    - Run an auto-masking algorithm to detect structure and define a cngi.image.region
    - Apply a pblimit based mask

    An existing deconvolution mask from img_dataset may either be included in the above, or ignored.

    The output is a region (array?) in the img_dataset containing the intersection of all above regions

    Returns
    -------
    img_dataset : xarray.core.dataset.Dataset
    """
