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

def feather(img_dataset_lowres, img_dataset_highres):
    """
    .. todo::
        This function is not yet implemented
    
    Feather two images together, based on restoring beam information stored in both.
    
    Output image = iFT( FT(lowres_image) + [1-FT(lowres_beam)] x FT(highres_image) )
    
    TBD : Do this for the entire image_set (psf, image) and updated restoring-beam information as well ?
    
    Returns
    -------
    img_dataset : xarray.core.dataset.Dataset
    """
