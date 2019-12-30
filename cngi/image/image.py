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
def rebin(ds, **kwargs):

    """
    .. todo::
        Check for and apply all masks and regions
        Accept arguments that control which DataArray is binned
        Improve performance when framework client has processes=True
    
    Re-bin an image in any spatial or spectral direction
    
    Parameters
    ----------
    ds : xarray Dataset
        input Image
    factor : int
        scaling factor for binning, Default=1
    axis : string
        dataset dimension upon which to rebin, Default='frequency'
    
    Returns
    -------
    xarray Dataset
        New Dataset
    """
    
    if 'factor' in kwargs.keys():
        factor = kwargs['factor']
    else:
        factor = 1
        
    if 'axis' in  kwargs.keys():
        axis = kwargs['axis']
        if axis in ds.dims:
            pass
        else:
            print("Requested axis not a dimension of input dataset.")
    else:
        print("Defaulting to rebinning dataset by frequency dimension")
        axis = 'frequency'

    # works best with threads
    # keep_attrs not implemented here yet (xarray/issues/3376)
    new_ds = ds.coarsen({axis:factor}).mean(keep_attrs=True)

    return new_ds



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
def moment(ds, **kwargs):
    """
    .. todo::
        Verify implemented moment codes
        Implement additional moment codes
    
    Collapse the cube into a moment by taking a linear combination of the individual planes
    
    Parameters
    ----------
    ds : xarray Dataset
        input Image
    axis : string
        specified axis along which to reduce for moment generation, Default='frequency'
    code : int
        number that selects which moment to calculate from the following list, Default=-1

        code=-1 - mean value of the spectrum  
        code=0  - integrated value of the spectrum  
        code=1  - intensity weighted coordinate; traditionally used to get ’velocity fields’
        code=2  - intensity weighted dispersion of the coordinate; traditionally used to get ’velocity dispersion’  
        code=3  - median of I  
        code=4  - median coordinate  
        code=5  - standard deviation about the mean of the spectrum  
        code=6  - root mean square of the spectrum  
        code=7  - absolute mean deviation of the spectrum  
        code=8  - maximum value of the spectrum  
        code=9  - coordinate of the maximum value of the spectrum  
        code=10 - minimum value of the spectrum  
        code=11 - coordinate of the minimum value of the spectrum

    Returns
    -------
    xarray Dataset
        New Dataset
    """

    # input parameter checking
    if 'axis' in kwargs.keys():
        axis = kwargs['axis']
    else:
        print("No axis specified."
              "Defaulting to reducing along frequency dimension")
        axis = 'frequency'

    if 'code' in kwargs.keys():
        code = int(kwargs['code'])
        assert code in range(-1,12), "Input to 'code' parameter must be between -1 and 11"
    else:
        print("No valid input code detected, assuming default (-1)")
        code = -1

    # moment calculation
    if code == -1:
        new_ds = ds.mean(dim=axis, keep_attrs=True)
    if code == 0:
        new_ds = ds.integrate(dim=axis, keep_attrs=True)
    if code == 1:
        new_ds = (ds.sum('frequency', keep_attrs=True) / 
                  (ds.integrate(dim=axis, keep_attrs=True)))
    if code == 8:
        new_ds = ds.max(dim=axis, keep_atrs=True)
    if code == 10:
        new_ds = ds.reduce(func=min, dim='frequency', keepdims=True)
    else:
        raise NotImplementedError(f"Moment code={code} is not yet supported")

    return new_ds



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



