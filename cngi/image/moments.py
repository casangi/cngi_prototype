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
from cngi.image import imageUtility as iu

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
    # moment: int array: a List of moments to compute
    if 'moment' in kwargs.keys():
        moment = kwargs['moment']
        assert np.min(moment) in range(-1,12), "Input to moment parameter must be between -1 and 11"
        assert np.max(moment) in range(-1,12)
    else:
        print("No valid input code detected, assuming default to calculate all image moments)")
        moment = np.arange(-1,12)

    # axis: string int: the moment axis: ra, dec, lat, long, spectral or stokes.
    if 'axis' in kwargs.keys():
        axis = kwargs['axis']
    else:
        print("No valid axis is specified, set default 'chan')")
        axis='chan'

    # chans: string, Channels to use. Default is to use all channels.
    if ('chans' in kwargs.keys()):
        chans = iu.selectedchannels(chans = kwargs['chans'],shapeLength = ds.dims['chan'])
    else:
        print("No channel is specified. Set to default -1 to use all channels")
        chans = np.arange(ds.dims['chan'])

    #This factor is related to the width (in world coordinate units) of a pixel along the moment axis
    # light speed in kilometer per second
    lightSpeed = 2.99792458*pow(10,5)
    f0 = float(ds.attrs['rest_frequency'].replace('hz', ''))
    v = (1 - ds.coords[axis] / f0) * lightSpeed
    deltaV = (ds.coords[axis].values[1]-ds.coords[axis].values[0])*lightSpeed / f0
    intensity=ds.IMAGE[:,:,chans,:]

    # moments calculation
    if -1 in moment or 0 in moment or 1 in moment or 2 in moment:
        #ds["MOMENTS_AVERAGE"]=intensity.mean(dim=axis)
        ds["MOMENTS_AVERAGE"] = intensity.sum(dim=axis) / intensity.shape[2]
        ds["MOMENTS_INTERGRATED"]=intensity.sum(dim=axis)*deltaV

        sum1 = 0
        for i in range(intensity.shape[2]):
            sum1 += intensity[:, :, i, :] *  v[i]
        ds["MOMENTS_WEIGHTED_COORD"] = sum1/ds["MOMENTS_INTERGRATED"]

        sum1=0
        for i in range(intensity.shape[2]):
            sum1 = intensity[:, :, i, :]*pow((v[i]-ds["MOMENTS_WEIGHTED_COORD"]),2)
        ds["MOMENTS_WEIGHTED_DISPERSION_COORD"] = np.sqrt(sum1 / ds["MOMENTS_INTERGRATED"])

    if 3 in moment:
        ds["MOMENTS_MEDIAN"] = intensity.median(dim=axis)
    if 4 in moment:
        ds["MOMENTS_MEDIAN_COORD"] = np.quantile(intensity.values, .25)
    if 5 in moment:

        sd = pow((intensity - intensity.mean(dim=axis)),2)
        standarddeviation = np.sqrt(sd.sum(dim=axis) / (intensity.shape[2]-1))
        ds["MOMENTS_STANDARD_DEVIATION"] = standarddeviation

        # The default xarray.std returns large difference between casa6 and CNGI
        # ds["MOMENTS_STANDARD_DEVIATION"] = intensity.std(dim=axis)
    if 6 in moment:
        ds["MOMENTS_RMS"] = np.sqrt((np.fabs(intensity * intensity)).mean(dim=axis))
    if 7 in moment:
        sd = np.fabs((intensity-intensity.mean(dim=axis)))
        absmeandev = sd.mean(dim=axis)
        ds["MOMENTS_ABS_MEAN_DEV"] = absmeandev
    if 8 in moment:
        ds["MOMENTS_MAXIMUM"] = intensity.max(dim=axis)
    # moments of maximum coordinate unit is km/m
    if 9 in moment:
        mcnparray = intensity.argmax(dim=axis).values.astype(np.float32)
        for i in range(intensity.shape[2]):
            mcnparray[mcnparray==i]=v[i]

        ds["MOMENTS_MAXIMUM_COORD"] = xa.DataArray(mcnparray,
                                                   coords=ds["MOMENTS_MAXIMUM"].coords,
                                                   dims=ds["MOMENTS_MAXIMUM"].dims)
    if 10 in moment:
        ds["MOMENTS_MINIMUM"] = intensity.min(dim=axis)
    # moments of maximum coordinate unit is km/m
    if 11 in moment:
        mcnparray = intensity.argmin(dim=axis).values.astype(np.float32)
        for i in range(intensity.shape[2]):
            mcnparray[mcnparray == i] = v[i]

        ds["MOMENTS_MINIMUM_COORD"] = xa.DataArray(mcnparray,
                                                   coords=ds["MOMENTS_MAXIMUM"].coords,
                                                   dims=ds["MOMENTS_MAXIMUM"].dims)

    return ds



