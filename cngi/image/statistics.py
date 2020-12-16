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

########################
from cngi.conversion import convert_image
from cngi.image import implot
from cngi.image import statistics
import os
import numpy as np
from astropy import units as u
from astropy.coordinates import Angle
from cngi.image import imageUtility as iu

def statistics(xds, **kwargs):
    """
    Generate statistics on specified image data contents

    Resulting data is placed in the attributes section of the dataset

    Parameters
    ----------
    xds : xarray.core.dataset.Dataset
        input Image Dataset
    dv : str
        name of data_var in xds to compute statistics on. Default is 'IMAGE'
    axes
    Cursor axes over which to evaluate statistics.

    listit
    Print stats and bounding box to logger?

    verbose
    Print additional, possibly useful, messages to logger?

    logfile
    Name of file to write fit results.

    append
    If logfile exists, append to it if True or overwrite it if False.

    algorithm
    Algorithm to use. Supported values are "biweight", "chauvenet", "classic", "fit-half", and "hinges-fences". Minimum match is supported.

    fence
    Fence value for hinges-fences. A negative value means use the entire data set (ie default to the "classic" algorithm). Ignored if algorithm is not "hinges-fences".

    center
    Center to use for fit-half. Valid choices are "mean", "median", and "zero". Ignored if algorithm is not "fit-half".

    lside
    For fit-half, use values <= center for real data if True? If False, use values >= center as real data. Ignored if algorithm is not "fit-half".

    zscore
    For chauvenet, this is the target maximum number of standard deviations data may have to be included. If negative, use  Chauvenet's criterion. Ignored if algorithm is not "chauvenet".

    maxiter
    For chauvenet, this is the maximum number of iterations to attempt. Iterating will stop when either this limit is reached, or the zscore criterion is met. If negative, iterate until the zscore criterion is met. Ignored if algorithm is not "chauvenet".

    clmethod
    Method to use for calculating classical statistics. Supported methods are "auto", "tiled", and "framework". Ignored if algorithm is not "classic".

    niter
    For biweight, this is the maximum number of iterations to attempt. Iterating will stop when either this limit is reached, or the convergence criterion is met. If negative, do a fast, simple computation (see description). Ignored if the algorithm is not "biweight".
        Returns
    -------
    xarray.core.dataset.Dataset
        output Image
    """

    if 'dv' in kwargs.keys():
        dv = int(kwargs['dv'])
    else:
        dv = 'IMAGE'

    # axis: string int: the moment axis: ra, dec, lat, long, spectral or stokes.
    # Cursor axes over which to evaluate statistics

    # axis: string int: the moment axis: ra, dec, lat, long, spectral or stokes.
    if 'axis' in kwargs.keys():
        axis = kwargs['axis']
    else:
        print("No valid axis is specified, set default 'chan')")
        axis = 'chan'

    # chans: string, Channels to use. Default is to use all channels.
    if ('chans' in kwargs.keys()):
        chans = iu.selectedchannels(chans=kwargs['chans'], shapeLength=xds.dims['chan'])
    else:
        chans = None

    # chans: string, Channels to use. Default is to use all channels.
    if ('pols' in kwargs.keys()):
       pols=kwargs['pols']
    else:
       pols = None

    if 'logfile' in kwargs.keys():
        logfile = kwargs['logfile']
        if 'append' in kwargs.keys():
            append = kwargs['logfile']
        else:
            append = False
    else:
        print("Not write statistics results into a logfile")
        logfile = False

    # algorithm "biweight", "chauvenet", "classic", "fit-half", and "hinges-fences".
    if 'algorithm' in kwargs.keys():
        algorithm = kwargs['algorithm']
    else:
        algorithm = 'classic'

    if algorithm == 'classic':
        if 'clmethod' in kwargs.keys():
            clmethod = kwargs['clmethod']
        else:
            clmethod = 'auto'

    if 'algorithm'  == 'hinges-fences':
       if 'fence' in kwargs.keys():
           fence = kwargs['fence']
       else:
           fence = -1

    if algorithm  == 'fit-half':
       if 'center' in kwargs.keys():
           center = kwargs['center']
       else:
           center = mean
       if 'lside' in kwargs.keys():
           lside = kwargs['lside']
       else:
           lside = True

    if algorithm == 'chauvenet':
        if 'zscore' in kwargs.keys():
            zscore = kwargs['zscore']
        else:
            zscore = -1
        if 'maxiter' in kwargs.keys():
            maxiter = kwargs['maxiter']
        else:
            maxiter = -1

    if algorithm == 'biweight':
        if 'niter' in kwargs.keys():
            niter = kwargs['niter']
        else:
            niter = -1

    if pols is not None:
        xda = xds.isel(pol=pols)

    if chans is not None:
       intensity = xds[dv][:,:,chans,:]
    else:
       intensity = xds[dv]

    # the number of unmasked points used
    nps = 1
    for i in intensity.shape:
        nps = nps * i

    # the sum of the pixel values
    sum = intensity.values.sum()

    # the sum of the squares of the pixel value
    sumsq = np.sum((intensity.values * intensity.values))

    # the mean of pixel values
    mean = intensity.values.mean()

    # the standard deviation about the mean
    sigma = intensity.values.std()

    # the root mean sqaure
    rms = np.sqrt(sumsq / nps)

    # minimum pixel value
    min = intensity.values.min()

    # maximum pixel value
    max = intensity.values.max()

    # the median pixel value
    median = np.median(intensity.values)

    # the median of the absolute deviations from the median
    medabsdevmed = np.median(np.abs(intensity.values - np.median(intensity.values)))

    # the first quartile
    q1 = np.quantile(intensity.values, .25)

    # the third quartile
    q3 = np.quantile(intensity.values, .75)

    # the inter-quartile range (if robust=T). Find the points which are 25% largest and 75% largest (the median is 50% largest).
    quartile = q3 - q1

    # the absolute pixel coordinate of the bottom left corner of the bounding box of the region of interest.
    # If ’region’ is unset, this will be the bottom left corner of the whole image.
    blc = [0] * len(intensity.dims)

    #  the formatted absolute world coordinate of the bottom left corner of the bounding box of the region of interest.
    blcf = getWCSvalueFromIndex(xds, blc)

    # trc - the absolute pixel coordinate of the top right corner of the bounding box of the region of interest.
    trc = [intensity.shape[0] - 1, intensity.shape[1] - 1, intensity.shape[2] - 1, intensity.shape[3] - 1]

    # trcf - the formatted absolute world coordinate of the top right corner of the bounding box of the region of interest.
    trcf = getWCSvalueFromIndex(xds, trc)

    # absolute pixel coordinate of minimum pixel value
    minIndex = np.where(intensity.values == np.amin(intensity.values))
    minPos = [minIndex[0][0], minIndex[1][0], minIndex[2][0], minIndex[3][0]]
    minPosf = getWCSvalueFromIndex(xds, minPos)

    # absolute pixel coordinate of maximum pixel value
    maxIndex = np.where(intensity.values == np.amax(intensity.values))
    maxPos = [maxIndex[0][0], maxIndex[1][0], maxIndex[2][0], maxIndex[3][0]]
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

    return xds, statisticsResults


def getWCSvalueFromIndex(xds, indexArray):
    results = [0] * len(indexArray)
    results[0] = Angle(xds.right_ascension.values[indexArray[0], indexArray[1]], u.radian).to_string(unit=u.hour,
                                                                                                     sep=(':'))
    results[1] = Angle(xds.declination.values[indexArray[0], indexArray[1]], u.radian).to_string(unit=u.deg, sep=('.'))
    results[2] = xds.chan.values[indexArray[2]]
    results[3] = xds.pol.values[indexArray[3]]
    return results;

