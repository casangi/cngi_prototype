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

import numpy as np


def statistics(xds, dv='IMAGE', algorithm='classic', clmethod='auto', fence=-1.0, center='mean', lside=True, zscore=-1.0, maxiter=1, niter=-1):
    """
    Generate statistics on specified image data contents

    Resulting data is placed in the attributes section of the dataset

    Parameters
    ----------
    xds : xarray.core.dataset.Dataset
       input Image Dataset
    dv : str
       name of data_var in xds to compute statistics on. Default is 'IMAGE'
    algorithm : str
       statistics algorithm to use. Choices are 'biweight', 'chauvenet', 'classic',
       'fit-half', and 'hinges-fences'. Default is 'classic'
    clmethod : str
       Method to use for calculating classical statistics. Supported methods are "auto",
       "tiled", and "framework". Ignored if algorithm is not "classic". Default is 'auto'
    fence : float
       Fence value for hinges-fences. A negative value means use the entire data set (ie
       default to the "classic" algorithm). Ignored if algorithm is not "hinges-fences".
       Default is -1.0
    center : str
       Center to use for fit-half. Valid choices are "mean", "median", and "zero". Ignored
       if algorithm is not "fit-half".  Default is 'mean'
    lside : bool
       For fit-half, use values <= center for real data if True. If False, use values >= center
       as real data. Ignored if algorithm is not "fit-half". Default is True
    zscore : float
       For chauvenet, this is the target maximum number of standard deviations data may have to
       be included. If negative, use Chauvenet's criterion. Ignored if algorithm is not "chauvenet".
       Default is -1.0
    maxiter : int
       For chauvenet, this is the maximum number of iterations to attempt. Iterating will stop when
       either this limit is reached, or the zscore criterion is met. If negative, iterate until the
       zscore criterion is met. Ignored if algorithm is not "chauvenet". Default is 1
    niter : int
        For biweight, this is the maximum number of iterations to attempt. Iterating will stop when
        either this limit is reached, or the convergence criterion is met. If negative, do a fast,
        simple computation (see description). Ignored if the algorithm is not "biweight".
        Default is -1
        
    Returns
    -------
    xarray.core.dataset.Dataset
        output Image
    """
    
    xds = xds.copy()
    
    assert dv in xds.data_vars, "axis not present in input image"
    assert algorithm in ["biweight", "chauvenet", "classic", "fit-half", "hinges-fences"], 'invalid algorithm parameter value'

    if algorithm == 'classic':
        assert clmethod in ["auto","tiled","framework"], 'invalid clmethod parameter'

    if algorithm == 'fit-half':
        assert center in ["mean", "median", "zero"], 'invalid center parameter value'
       
    
    intensity = xds[dv]
    
    # the number of unmasked points used
    # don't use a big loop, vectorized mathematics is faster
    #nps = 1
    #for i in intensity.shape:
    #    nps = nps * i
    nps = (intensity != np.nan).astype(int).sum().values

    # the sum of the pixel values
    sum = intensity.sum().values

    # the sum of the squares of the pixel value
    sumsq = np.sum(intensity * intensity).values

    # the mean of pixel values
    mean = intensity.mean().values

    # the standard deviation about the mean
    sigma = intensity.std().values

    # the root mean sqaure
    rms = np.sqrt(sumsq / nps)

    # minimum pixel value
    min = intensity.min().values

    # maximum pixel value
    max = intensity.max().values

    # the median pixel value
    median = intensity.median(dim=['l','m','time','pol']).values

    # the median of the absolute deviations from the median
    medabsdevmed = np.abs(intensity - np.median(intensity)).median(dim=['l','m','time','pol']).values

    # the first quartile
    q1 = intensity.chunk({'chan': -1}).quantile(0.25).values
    

    # the third quartile
    q3 = intensity.chunk({'chan': -1}).quantile(0.75).values

    # the inter-quartile range (if robust=T). Find the points which are 25% largest and 75% largest (the median is 50% largest).
    quartile = (q3 - q1)

    # the absolute pixel coordinate of the bottom left corner of the bounding box of the region of interest.
    # If ’region’ is unset, this will be the bottom left corner of the whole image.
    blc = [0] * len(intensity.dims)

    #  the formatted absolute world coordinate of the bottom left corner of the bounding box of the region of interest.
    blcf = getWCSvalueFromIndex(xds, blc)

    # trc - the absolute pixel coordinate of the top right corner of the bounding box of the region of interest.
    trc = list(np.array(intensity.shape)-1)

    # trcf - the formatted absolute world coordinate of the top right corner of the bounding box of the region of interest.
    trcf = getWCSvalueFromIndex(xds, trc)

    # absolute pixel coordinate of minimum pixel value
    minIndex = np.where(intensity == np.amin(intensity))
    minPos = [minIndex[0][0], minIndex[1][0], minIndex[2][0], minIndex[3][0], minIndex[4][0]]
    minPosf = getWCSvalueFromIndex(xds, minPos)

    # absolute pixel coordinate of maximum pixel value
    maxIndex = np.where(intensity == np.amax(intensity))
    maxPos = [maxIndex[0][0], maxIndex[1][0], maxIndex[2][0], maxIndex[3][0], maxIndex[4][0]]
    maxPosf = getWCSvalueFromIndex(xds, maxPos)

    statisticsResults = {
        "blc": blc,
        "blcf": blcf,
        # "flux": flux,
        "max": max,
        "maxpos": maxPos,
        "maxposf": maxPosf,
        "mean": mean,
        "medabsdevmed": medabsdevmed,
        "median": median,
        "min": min,
        "minpos": minPos,
        "minposf": minPosf,
        "npts": nps,
        "q1": q1,
        "q3": q3,
        "quartile": quartile,
        "rms": rms,
        "sigma": sigma,
        "sum": sum,
        "sumsq": sumsq,
        "trc": trc,
        "trcf": trcf
    }

    return xds.assign_attrs({'statistics':statisticsResults})



#####################################
def getWCSvalueFromIndex(xds, indexArray):
    from astropy import units as u
    from astropy.coordinates import Angle

    results = [0] * len(indexArray)
    results[0] = Angle(xds.right_ascension[indexArray[0], indexArray[1]].values, u.radian).to_string(unit=u.hour, sep=(':'))
    results[1] = Angle(xds.declination[indexArray[0], indexArray[1]].values, u.radian).to_string(unit=u.deg, sep=('.'))
    results[2] = xds.time[indexArray[2]].values
    results[2] = xds.chan[indexArray[3]].values
    results[3] = xds.pol[indexArray[4]].values
    return results;

