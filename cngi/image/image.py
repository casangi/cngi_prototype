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
    ds : xarray Dataset
        input Image
    
    Returns
    -------
    xarray Dataset
        New Dataset
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
    ds : xarray Dataset
        input Image
    
    Returns
    -------
    xarray Dataset
        New Dataset
    """
    return True




########################
def rebin(ds):
    """
    .. todo::
        This function is not yet implemented
    
    Re-bin an image in any spatial or spectral direction
    
    Parameters
    ----------
    ds : xarray Dataset
        input Image
    
    Returns
    -------
    xarray Dataset
        New Dataset
    """
    return True



########################
def contsub(ds):
    """
    .. todo::
        This function is not yet implemented
    
    Continuum subtraction of an image cube
    
    Perform a polynomial baseline fit to the specified channels from an image and subtract it from all channels
    
    Parameters
    ----------
    ds : xarray Dataset
        input Image
    
    Returns
    -------
    xarray Dataset
        New Dataset
    """
    return True



########################
def moment(ds):
    """
    .. todo::
        This function is not yet implemented
    
    Collapse the cube in to a moment by taking a linear combination of the individual planes
    
    Parameters
    ----------
    ds : xarray Dataset
        input Image
    
    Returns
    -------
    xarray Dataset
        New Dataset
    """
    return True



########################
def smooth(ds):
    """
    .. todo::
        This function is not yet implemented
    
    Smooth data along n-dimensions of the image cube
    
    Parameters
    ----------
    ds : xarray Dataset
        input Image
    
    Returns
    -------
    xarray Dataset
        New Dataset
    """
    return True



########################
def specfit(ds):
    """
    .. todo::
        This function is not yet implemented
    
    Perform polynomial, gaussian and lorentzian spectral line fits in the image cube
    
    Parameters
    ----------
    ds : xarray Dataset
        input Image
    
    Returns
    -------
    xarray Dataset
        New Dataset
    """
    return True



########################
def spxfit(ds):
    """
    .. todo::
        This function is not yet implemented
    
    Fit a power logarithmic polynomial to pixel values along specified axis
    
    Parameters
    ----------
    ds : xarray Dataset
        input Image
    
    Returns
    -------
    xarray Dataset
        New Dataset
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
    ds : xarray Dataset
        input Image
    
    Returns
    -------
    xarray Dataset
        New Dataset
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
    ds : xarray Dataset
        input Image
    
    Returns
    -------
    xarray Dataset
        New Dataset
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
    ds : xarray Dataset
        input Image
    
    Returns
    -------
    xarray Dataset
        New Dataset
    """
    return True




########################
def mask(ds):
    """
    .. todo::
        This function is not yet implemented
    
    Create an n-dim mask in the image
    
    Parameters
    ----------
    ds : xarray Dataset
        input Image
    
    Returns
    -------
    xarray Dataset
        New Dataset
    """
    return True




########################
def region(ds):
    """
    .. todo::
        This function is not yet implemented
    
    Create an n-dim region in the image
    
    Parameters
    ----------
    ds : xarray Dataset
        input Image
    
    Returns
    -------
    xarray Dataset
        New Dataset
    """
    return True

