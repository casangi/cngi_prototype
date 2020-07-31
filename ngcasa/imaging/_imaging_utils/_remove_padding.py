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
