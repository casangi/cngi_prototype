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
import xarray


##################################################################
# copies mxds and replaces specified vis partition with new xds
def mxds_copier(mxds, vis, xds):
    txds = mxds.copy()
    txds.attrs[vis] = xds
    return txds


##################################################################
# takes a list of visibility xarray datasets and packages them as a dataset of datasets
# xds_list is a list of tuples (name, xds)
def vis_xds_packager(xds_list):
    mxds = xarray.Dataset(attrs=dict(xds_list))
    
    coords = {}
    if 'ANTENNA' in mxds.attrs:
        coords['antenna_ids'] = mxds.ANTENNA.antenna_id.values
        coords['antennas'] = xarray.DataArray(mxds.ANTENNA.NAME.values, dims=['antenna_ids'])
    if 'FIELD' in mxds.attrs:
        coords['field_ids'] = mxds.FIELD.field_id.values
        coords['fields'] = xarray.DataArray(mxds.FIELD.NAME.values, dims=['field_ids'])
    if 'FEED' in mxds.attrs:
        coords['feed_ids'] = mxds.FEED.feed_id.values
    if 'OBSERVATION' in mxds.attrs:
        coords['observation_ids'] = mxds.OBSERVATION.observation_id.values
        coords['observations'] = xarray.DataArray(mxds.OBSERVATION.PROJECT.values, dims=['observation_ids'])
    if 'POLARIZATION' in mxds.attrs:
        coords['polarization_ids'] = mxds.POLARIZATION.d0.values
    if 'SOURCE' in mxds.attrs:
        coords['source_ids'] = mxds.SOURCE.source_id.values
        coords['sources'] = xarray.DataArray(mxds.SOURCE.NAME.values, dims=['source_ids'])
    if 'SPECTRAL_WINDOW' in mxds.attrs:
        coords['spw_ids'] = mxds.SPECTRAL_WINDOW.d0.values
    if 'STATE' in mxds.attrs:
        coords['state_ids'] = mxds.STATE.state_id.values
    
    mxds = mxds.assign_coords(coords)
    
    return mxds
