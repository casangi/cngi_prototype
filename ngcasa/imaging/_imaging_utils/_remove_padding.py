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

#ducting - code is complex and might fail after some time if parameters is wrong (time waisting). Sensable values are also checked. Gives printout of all wrong parameters. Dirty images alone has x parametrs.


def _remove_padding(image_dask_array,image_size):
    #Check that image_size < image_size
    #Check parameters
    
    import numpy as np
    
    image_size_padded= np.array(image_dask_array.shape[0:2])
    start_xy = (image_size_padded // 2 - image_size // 2)
    end_xy = start_xy + image_size
    
    image_dask_array = image_dask_array[start_xy[0]:end_xy[0], start_xy[1]:end_xy[1]]
    return image_dask_array
