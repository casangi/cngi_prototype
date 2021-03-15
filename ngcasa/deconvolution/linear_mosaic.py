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

def linear_mosaic(img_dataset, img_mosaic):
    """
    .. todo::
        This function is not yet implemented
    
    Construct a linear mosaic as a primary-beam weighted sum of a set of input images.
    Individual images are re-sampled onto a larger image grid and summed.
       
    Assume flat-noise normalization for the inputs.  ( TBD : Or flatsky? )
    
    Output image :  sum( input_images ) / sum ( input_pbs )
    
    TBD :
    
    This requires some sort of merging of img_datasets.  CNGI demo on how to append/add images to an image_set and ensure
    that meta-data are consistent?
    
    Returns
    -------
    img_dataset : xarray.core.dataset.Dataset
    """
