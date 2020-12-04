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

from casatasks import imstat
from cngi.conversion import convert_image
from cngi.image import implot
from cngi.image import statistics
import os
import numpy as np
from astropy import units as u
from astropy.coordinates import Angle



def statistics(xds, dv='IMAGE', mode='classic'):
    """
    Generate statistics on specified image data contents
    
    Resulting data is placed in the attributes section of the dataset

    Parameters
    ----------
    xds : xarray.core.dataset.Dataset
        input Image Dataset
    dv : str
        name of data_var in xds to compute statistics on. Default is 'IMAGE'
    mode : str
        algorithm mode to use ('classic', 'fit-half', 'hinges-fences', 'chauvenet', 'biweight'). Default is 'classic'
    
    Returns
    -------
    xarray.core.dataset.Dataset
        output Image
    """

    intensity=xds.IMAGE
    arraySize = len(intensity.dims)
    # the number of unmasked points used
    nps=1
    for i in intensity.shape:
        nps=nps * i

    # the sum of the pixel values
    sum = intensity.values.sum()

    # the sum of the squares of the pixel value
    sumsq = np.sum((intensity.values * intensity.values))

    # the mean of pixel values
    mean = intensity.values.mean()

    # the standard deviation about the mean
    sigma = intensity.values.std()

    # the root mean sqaure
    rms = np.sqrt(sumsq/nps)

    # minimum pixel value
    min = intensity.values.min()

    # maximum pixel value
    max=intensity.values.max()

    # the median pixel value
    median = np.median(intensity.values)

    # the median of the absolute deviations from the median
    medabsdevmed = np.median(np.abs(intensity.values-np.median(intensity.values)))

    # the first quartile
    q1 = np.quantile(intensity.values,.25)

    # the third quartile
    q3 = np.quantile(intensity.values,.75)

    # the inter-quartile range (if robust=T). Find the points which are 25% largest and 75% largest (the median is 50% largest).
    quartile = q3 - q1

    # the absolute pixel coordinate of the bottom left corner of the bounding box of the region of interest.
    # If ’region’ is unset, this will be the bottom left corner of the whole image.
    blc = [0] * len(intensity.dims)

    #  the formatted absolute world coordinate of the bottom left corner of the bounding box of the region of interest.
    blcf =getWCSvalueFromIndex(xds,blc)

    # trc - the absolute pixel coordinate of the top right corner of the bounding box of the region of interest.
    trc = [intensity.shape[0]-1,intensity.shape[1]-1, intensity.shape[2]-1, intensity.shape[3]-1]

    # trcf - the formatted absolute world coordinate of the top right corner of the bounding box of the region of interest.
    trcf = getWCSvalueFromIndex(xds,trc)

    # absolute pixel coordinate of minimum pixel value
    minIndex = np.where(intensity.values == np.amin(intensity.values))
    minPos = [minIndex[0][0], minIndex[1][0], minIndex[2][0], minIndex[3][0]]
    minPosf = getWCSvalueFromIndex(xds,minPos)

    # absolute pixel coordinate of maximum pixel value
    maxIndex = np.where(intensity.values == np.amax(intensity.values))
    maxPos = [maxIndex[0][0], maxIndex[1][0], maxIndex[2][0], maxIndex[3][0]]
    maxPosf = getWCSvalueFromIndex(xds,maxPos)

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

    # formatted string of the world coordinate of the minimum pixel value
    #minposf

    # formatted string of the world coordinate of the maximum pixel value
    #maxposf

    return xds

def getWCSvalueFromIndex(xds,indexArray):
    results = [0]*len(indexArray)
    results[0] = Angle(xds.right_ascension.values[indexArray[0], indexArray[1]], u.radian).to_string(unit=u.hour, sep=(':'))
    results[1] = Angle(xds.declination.values[indexArray[0], indexArray[1]], u.radian).to_string(unit=u.deg,  sep=('.'))
    results[2] = xds.chan.values[indexArray[2]]
    results[3] = xds.pol.values[indexArray[3]]
    return results;

