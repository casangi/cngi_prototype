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

def statistics(xds, dv='IMAGE', name='statistics', compute=False):
    """
    Generate statistics on specified image data contents

    Resulting data is placed in the attributes section of the dataset

    Parameters
    ----------
    xds : xarray.core.dataset.Dataset
       input Image Dataset
    dv : str
       name of data_var in xds to compute statistics on. Default is 'IMAGE'
    name : str
       name of the attribute in xds to hold statistics dictionary.  Default is 'statistics'
    compute : bool
       execute the DAG to compute the statistics. Default False returns lazy DAG
       (statistics can then be retrieved via xds.<name>.<key>.values)
       
    Returns
    -------
    xarray.core.dataset.Dataset
        output Image
    """
    import numpy as np
    import dask.array as da
   
    assert dv in xds.data_vars, "axis not present in input image"
        
    intensity = xds[dv]
    
    # the number of unmasked points used
    # don't use a big loop, vectorized mathematics is faster
    #nps = 1
    #for i in intensity.shape:
    #    nps = nps * i
    nps = (intensity != np.nan).astype(int).sum() #.values

    # the sum of the pixel values
    sum = intensity.sum()
    
    # the sum of the squares of the pixel value
    sumsq = (intensity * intensity).sum()
    
    # the mean of pixel values
    mean = intensity.mean()
    
    # the standard deviation about the mean
    sigma = intensity.std()

    # the root mean sqaure
    rms = (sumsq / nps)**0.5

    # minimum pixel value
    min = intensity.min()

    # maximum pixel value
    max = intensity.max()

    # the median pixel value
    median = intensity.median(dim=['l','m','chan','pol'])[0]  # one median. not median array.
    #median = np.median(intensity)

    # the median of the absolute deviations from the median    # median value, not median array for each channel
    medabsdevmed = np.abs(intensity - intensity.median(dim=['l','m','chan','pol'])).median(dim=['l','m','chan','pol'])[0]
    #medabsdevmed = np.median(np.abs(intensity - np.median(intensity)))

    # the first quartile
    q1 = intensity.chunk({'chan': -1}).quantile(0.25)
    
    # the third quartile
    q3 = intensity.chunk({'chan': -1}).quantile(0.75)

    # the inter-quartile range (if robust=T). Find the points which are 25% largest and 75% largest (the median is 50% largest).
    quartile = (q3 - q1)

    # the absolute pixel coordinate of the bottom left corner of the bounding box of the region of interest.
    # If ’region’ is unset, this will be the bottom left corner of the whole image.
    blc = da.from_array([0] * len(intensity.dims))

    #  the formatted absolute world coordinate of the bottom left corner of the bounding box of the region of interest.
    blcf = getWCSvalueFromIndex(xds, blc, compute)

    # trc - the absolute pixel coordinate of the top right corner of the bounding box of the region of interest.
    trc = da.from_array(intensity.shape)-1

    # trcf - the formatted absolute world coordinate of the top right corner of the bounding box of the region of interest.
    trcf = getWCSvalueFromIndex(xds, trc, compute)

    # absolute pixel coordinate of minimum pixel value
    #minIndex = np.where(intensity == np.amin(intensity))
    #minPos = [minIndex[0][0], minIndex[1][0], minIndex[2][0], minIndex[3][0], minIndex[4][0]]
    minPos = da.unravel_index(intensity.argmin().data, intensity.shape)
    minPosf = getWCSvalueFromIndex(xds, minPos, compute)

    # absolute pixel coordinate of maximum pixel value
    #maxIndex = np.where(intensity == np.amax(intensity))
    #maxPos = [maxIndex[0][0], maxIndex[1][0], maxIndex[2][0], maxIndex[3][0], maxIndex[4][0]]
    maxPos = da.unravel_index(intensity.argmax().data, intensity.shape)
    maxPosf = getWCSvalueFromIndex(xds, maxPos, compute)

    if compute:
        sum = sum.values.item()
        sumsq = sumsq.values.item()
        mean = mean.values.item()
        sigma = sigma.values.item()
        rms = rms.values.item()
        min = min.values.item()
        max = max.values.item()
        median = median.values.item()
        medabsdevmed = medabsdevmed.values.item()
        q1 = q1.values.item()
        q3 = q3.values.item()
        quartile = quartile.values.item()
        trc = trc.compute()
        blc = blc.compute()
        nps = nps.values.item()
        minPos = [mm.compute() for mm in minPos]
        maxPos = [mm.compute() for mm in maxPos]

    statisticsResults = {
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
        "blc": blc,
        "blcf": blcf,
        "trc": trc,
        "trcf": trcf
    }

    return xds.assign_attrs({name:statisticsResults})



#####################################
def getWCSvalueFromIndex(xds, ia, compute):
    from astropy import units as u
    from astropy.coordinates import Angle
    
    if compute:
        results = []
        results += [Angle(xds.right_ascension[ia[0], ia[1]].values, u.radian).to_string(unit=u.hour, sep=(':'))]
        results += [Angle(xds.declination[ia[0], ia[1]].values, u.radian).to_string(unit=u.deg, sep=('.'))]
        results += [xds.time[ia[2]].values.astype(str)]
        results += [xds.chan[ia[3]].values.item()]
        results += [xds.pol[ia[4]].values.item()]
    else:
        results = [xds.right_ascension[ia[0], ia[1]], xds.declination[ia[0], ia[1]], xds.time[ia[2]], xds.chan[ia[3]], xds.pol[ia[4]]]
        
    return results

