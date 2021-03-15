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
        coords['spw_ids'] = mxds.SPECTRAL_WINDOW.spectral_window_id.values
    if 'STATE' in mxds.attrs:
        coords['state_ids'] = mxds.STATE.state_id.values
    
    mxds = mxds.assign_coords(coords)
    
    return mxds
