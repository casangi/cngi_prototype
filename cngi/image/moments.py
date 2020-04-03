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
def moment(ds, **kwargs):
    """
    .. todo::
        This function is not yet implemented
    
    Collapse an n-dimensional image cube into a moment by taking a linear combination of individual planes
    
    .. note::
        This implementation still needs to implement additional moment codes, and verify behavior of implemented moment codes.
    
    Parameters
    ----------
    ds : xarray.core.dataset.Dataset
        input Image
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

    # moment calculation
    if code == -1:
        new_ds = ds.mean(dim=axis, keep_attrs=True)
    if code == 0:
        new_ds = xds.sum(dim='chan', keep_attrs=True)
    if code == 1:
        new_ds = (ds.sum('chan', keep_attrs=True) /
                  ds.integrate(dim=axis, keep_attrs=True))
    if code == 8:
        new_ds = ds.max(dim=axis, keep_atrs=True)
    if code == 10:
        new_ds = ds.reduce(func=min, dim='chan', keepdims=True)
    else:
        raise NotImplementedError(f"Moment code={code} is not yet supported")

    return new_ds



