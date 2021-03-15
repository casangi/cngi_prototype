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

def predict_modelvis_image(img_dataset, vis_dataset, grid_parms, storage_parms):
    """
    .. todo::
        This function is not yet implemented
        
    Predict model visibilities from an input model image cube (units Jy/pixel) using a pre-specified gridding convolution function cache.
    
    Save the model visibilities in arr_name (default = 'MODEL')
    
    Optionally overwrite the model or add to existing model (incremental=T)
    
    (A input cube with 1 channel is a continuum image (nterms=1))

    Returns
    -------
    vis_dataset : xarray.core.dataset.Dataset
    """
    #ngcasa.imaging._normalize(direction='forward') # Apply PB models to go to flat-sky
    #cngi.image.stokes_to_corr()
    #cngi.image.fourier_transform()
    #ngcasa.imaging._degrid()
