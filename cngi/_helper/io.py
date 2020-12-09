#   Copyright 2020 AUI, Inc. Washington DC, USA
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

#################################
# Helper File
#
# Not exposed in API
#
#################################


##################################################################
# takes a list of visibility xarray datasets and packages them as a dataset of datasets
# xds_list is a list of tuples (name, xds)
def vis_xds_packager(xds_list):
    import xarray

    mxds = xarray.Dataset(attrs=dict(xds_list))
    
    coords = {}
    if 'ANTENNA' in mxds.attrs:
        coords['antennas'] = mxds.ANTENNA.NAME.values
    if 'FIELD' in mxds.attrs:
        coords['fields'] = mxds.FIELD.NAME.values
    if 'FEED' in mxds.attrs:
        coords['feeds'] = mxds.FEED.FEED_ID.values
    if 'OBSERVATION' in mxds.attrs:
        coords['observations'] =  mxds.OBSERVATION.PROJECT.values
    if 'POLARIZATION' in mxds.attrs:
        coords['polarizations'] = mxds.POLARIZATION.d0.values
    if 'PROCESSOR' in mxds.attrs:
        coords['processors'] = mxds.PROCESSOR.d0.values
    if 'SOURCE' in mxds.attrs:
        coords['sources'] = mxds.SOURCE.NAME.values
    if 'SPECTRAL_WINDOW' in mxds.attrs:
        coords['spws'] = mxds.SPECTRAL_WINDOW.d0.values
    if 'STATE' in mxds.attrs:
        coords['states'] = mxds.STATE.d0.values
    
    mxds = mxds.assign_coords(coords)
    
    return mxds
