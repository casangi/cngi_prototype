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



########################
def regrid(ds):
    """
    .. todo::
        This function is not yet implemented
    
    Regrid one image on to the coordinate system of another
    
    Parameters
    ----------
    ds : xarray.core.dataset.Dataset
        input Image
    
    Returns
    -------
    xarray.core.dataset.Dataset
        output Image
    """
    return True



########################
def reframe(ds):
    """
    .. todo::
        This function is not yet implemented
    
    Change the velocity system of an image
    
    Parameters
    ----------
    ds : xarray.core.dataset.Dataset
        input Image
    
    Returns
    -------
    xarray.core.dataset.Dataset
        output Image
    """
    return True


########################
def contsub(ds):
    """
    .. todo::
        This function is not yet implemented
        
    .. note::
        Potentially duplicates functionality of :py:func:`image.spxfit` and :py:func:`image.specfit`
    
    Continuum subtraction of an image cube
    
    Perform a polynomial baseline fit to the specified channels from an image and subtract it from all channels
    
    Parameters
    ----------
    ds : xarray.core.dataset.Dataset
        input Image
    
    Returns
    -------
    xarray.core.dataset.Dataset
        output Image
    """
    return True



########################
def specfit(ds):
    """
    .. todo::
        This function is not yet implemented
    
    .. note::
        Potentially duplicates functionality of :py:func:`image.contsub` and :py:func:`image.spxfit`

    Perform polynomial, gaussian and lorentzian spectral line fits in the image cube
    
    Parameters
    ----------
    ds : xarray.core.dataset.Dataset
        input Image
    
    Returns
    -------
    xarray.core.dataset.Dataset
        output Image
    """
    return True



########################
def spxfit(ds):
    """
    .. todo::
        This function is not yet implemented
    
    .. note::
        Potentially duplicates functionality of :py:func:`image.contsub` and :py:func:`image.specfit`

    Fit a power logarithmic polynomial to pixel values along specified axis
    
    Parameters
    ----------
    ds : xarray.core.dataset.Dataset
        input Image
    
    Returns
    -------
    xarray.core.dataset.Dataset
        output Image
    """
    return True



########################
def specflux(ds):
    """
    .. todo::
        This function is not yet implemented
    
    Calculate the flux as a function of frequency and velocity over the selected region
    
    Parameters
    ----------
    ds : xarray.core.dataset.Dataset
        input Image
    
    Returns
    -------
    xarray.core.dataset.Dataset
        output Image
    """
    return True



########################
def rmfit(ds):
    """
    .. todo::
        This function is not yet implemented
    
    Generate the rotation measure by performing a least square fit with Stokes Q and U axes
    
    Parameters
    ----------
    ds : xarray.core.dataset.Dataset
        input Image
    
    Returns
    -------
    xarray.core.dataset.Dataset
        output Image
    """
    return True



########################
def ellipsefit(ds):
    """
    .. todo::
        This function is not yet implemented
    
    Fit one or more elliptical gaussian components on an image region
    
    Parameters
    ----------
    ds : xarray.core.dataset.Dataset
        input Image
    
    Returns
    -------
    xarray.core.dataset.Dataset
        output Image
    """
    return True



