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


#############################################
def recalculateuvw(xds, field=None, refcode=None, reuse=True, phasecenter=None):
    """
    .. todo::
        This function is not yet implemented

    Recalulate UVW and shift data to new phase center

    Parameters
    ----------
    xds : xarray.core.dataset.Dataset
        input Visibility Dataset
    field : int
        fields to operate on. None = all
    refcode : str
        reference frame to convert UVW coordinates to
    reuse : bool
        base UVW calculation on the old values
    phasecenter : float
        direction of new phase center. None = no change

    Returns
    -------
    xarray.core.dataset.Dataset
        New Visibility Dataset with updated data
    """
    return {}

