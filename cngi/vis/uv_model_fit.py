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

#####################################################
def uv_model_fit(mxds, vis, niter=5, comptype='p', sourcepar=[1, 0, 0], varypar=[]):
    """
    .. todo::
        This function is not yet implemented

    Fit simple analytic source component models directly to visibility data

    Parameters
    ----------
    mxds : xarray.core.dataset.Dataset
        input multi-xarray Dataset with global data
    vis : str
        visibility partition in the mxds to use
    niter : int
        number of fitting iteractions to execute
    comptype : str
        component type (p=point source, g=ell. gauss. d=ell. disk)
    sourcepar : list
        starting fuess (flux, xoff, yoff, bmajaxrat, bpa)
    varypar : list
        parameters that may vary in the fit

    Returns
    -------
    xarray.core.dataset.Dataset
        New output multi-xarray Dataset with global data
    """
    return {}
