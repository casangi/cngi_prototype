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


##################################################
def uvcontsub(xds, field=None, fitspw=None, combine=None, solint='int', fitorder=0):
    """
    .. todo::
        This function is not yet implemented

    Estimate continuum emission and subtract it from visibilities

    Parameters
    ----------
    xds : xarray.core.dataset.Dataset
        input Visibility Dataset
    field : int
        field selection. If None, use all fields
    fitspw : int
        spw:channel selection for fitting
    combine : str
        data axis to combine for the continuum
    solint : str
        continuum fit timescale
    fitorder : int
        polynomial order for the fits

    Returns
    -------
    xarray.core.dataset.Dataset
        New Visibility Dataset with updated data
    """
    return {}

