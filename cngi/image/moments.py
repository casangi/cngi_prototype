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

import xarray as xa
import numpy as np

xa.set_options(keep_attrs=True)

def moments(ds, **kwargs):
    """
    Collapse an n-dimensional image cube into a moment by taking a linear combination of individual planes
    
    .. note::
        This implementation still needs to implement additional moment codes, and verify behavior of implemented moment codes.
    
    Parameters
    ----------
    ds : xarray.core.dataset.Dataset
        input Image Dataset
    axis : str, optional
        specified axis along which to reduce for moment generation, Default='chan'
    code : int, optional
        number that selects which moment to calculate from the following list

        -1 - mean value of the spectrum (default)

        0  - integrated value of the spectrum  

        1  - intensity weighted coordinate; traditionally used to get ’velocity fields’

        2  - intensity weighted dispersion of the coordinate; traditionally used to get ’velocity dispersion’  

        3  - median of I  

        4  - median coordinate  

        5  - standard deviation about the mean of the spectrum  

        6  - root mean square of the spectrum  

        7  - absolute mean deviation of the spectrum  

        8  - maximum value of the spectrum  

        9  - coordinate of the maximum value of the spectrum  

        10 - minimum value of the spectrum  

        11 - coordinate of the minimum value of the spectrum

    **kwargs
        Arbitrary keyword arguments

    Returns
    -------
    xarray.core.dataset.Dataset
        output Image
    """

    # input parameter checking
    if 'axis' in kwargs.keys():
        axis = kwargs['axis']
    else:
        print("No axis specified."
              "Defaulting to reducing along frequency dimension")
        axis = 'chan'

    if 'code' in kwargs.keys():
        code = int(kwargs['code'])
        assert code in range(-1,12), "Input to 'code' parameter must be between -1 and 11"
    else:
        print("No valid input code detected, assuming default (-1)")
        code = -1

    #This factor is related to the width (in world coordinate units) of a pixel along the moment axis
    #todo: need to find out how casa6 does
    channelSingleFactor=0.49100335
    intensity=ds.IMAGE
    code=kwargs['moments']
    deltachan= ds.coords['chan'].values[1]-ds.coords['chan'].values[0]
        # moment calculation
    if -1 in code or 0 in code or 1 in code or 2 in code:
        average = intensity.mean(dim='chan')
        ds["MOMENTS_AVERAGE"]=average.sel(pol=1)
        ds["MOMENTS_INTERGRATED"]=average.sel(pol=1)*channelSingleFactor*intensity.shape[2]

        sum1 = 0
        for i in range(intensity.shape[2]):
            sum1 += intensity[:, :, i, :] * i * deltachan
        mo = deltachan * intensity.sum(dim='chan')
        intensityweightedcoor = sum1/mo
        ds["MOMENTS_WEIGHTED_COORD"] = intensityweightedcoor.sel(pol=1)

        sum1=0
        for i in range(intensity.shape[2]):
            temp= i*deltachan-intensityweightedcoor
            temp= temp*temp
            sum1 = intensity[:, :, i, :]*temp
        ds["MOMENTS_WEIGHTED_DISPERSION_COORD"] = (sum1 / mo).sel(pol=1)

    if 3 in code:
        ds["MOMENTS_MEDIAN"] = intensity.median(dim='chan').sel(pol=1)
    if 4 in code:
        mediacoordinate = intensity.median(dim='chan')
        ds["MOMENTS_MEDIAN_COORD"] = mediacoordinate
    if 5 in code:
        '''
        sd = (intensity - average) * (intensity - average)
        standarddeviation = np.sqrt(sd.sum(dim='chan') / (intensity.shape[2]-1))
        ds["MOMENTS_STANDARD_DEVIATION"] = standarddeviation.sel(pol=1)
        '''
        ds["MOMENTS_STANDARD_DEVIATION"] = intensity.std(dim='chan').sel(pol=1)
    if 6 in code:
        ds["MOMENTS_RMS"] = np.sqrt((np.fabs(intensity * intensity)).mean(dim='chan')).sel(pol=1)
    if 7 in code:
        average = intensity.mean(dim='chan')
        sd = np.fabs((intensity-average))
        absmeandev = sd.mean(dim='chan')
        ds["MOMENTS_ABS_MEAN_DEV"] = absmeandev.sel(pol=1)
    if 8 in code:
        ds["MOMENTS_MAXIMUM"] = intensity.max(dim='chan').sel(pol=1)
    if 9 in code:
        ds["MOMENTS_MAXIMUM_COORD"] = intensity.argmax(dim='chan').sel(pol=1)
    if 10 in code:
        ds["MOMENTS_MINIMUM"] = intensity.min(dim='chan').sel(pol=1)
    if 11 in code:
        ds["MOMENTS_MINIMUM_COORD"] = intensity.argmin(dim='chan').sel(pol=1)
    #else:
    #   raise NotImplementedError(f"Moments code={code} is not yet supported")

    return ds



