#   Copyright 2021 AUI, Inc. Washington DC, USA
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

import xarray as xr

def _remap_coord(xds : xr.Dataset, coord_name : str, map_func) -> xr.Dataset:
    # get the new values
    old_vals = xds[coord_name].values
    new_vals = [map_func(x) for x in old_vals]

    # assign the coordinate
    if (coord_name in xds.dims):
        # assign as dimensional coordinate
        ret = xds.drop_vars([ coord_name ])
        ret = ret.assign_coords({ coord_name: new_vals })
    else:
        # assign as non-dimensional coordinate
        ret = xds.drop_vars([ coord_name ])
        coord_val = xr.DataArray(new_vals, dims=xds[coord_name].dims)
        ret = ret.assign({ coord_name: coord_val }) # creates coord_name as a data_var
        ret = ret.set_coords([ coord_name ]) # promote the new data_var "coord_name" to be a coordinate
    return ret

def _remap_data_var(xds : xr.Dataset, var_name : str, map_func) -> xr.Dataset:
    import numpy as np
    def mb(array):
        vals = array.values
        if (len(vals) > 0):
            vals = np.vectorize(map_func)(vals)
        return xr.DataArray(data=vals, coords=array.coords, dims=array.dims, name=array.name, attrs=array.attrs)
    var_val = xds[var_name].map_blocks(mb)
    return xds.assign({var_name: var_val})

def append_xds_subtable(sub0 : xr.Dataset, sub1 : xr.Dataset, relational_ids_map=None):
    """ Append the given subtable sub1 to a new subtable based on sub0. pseudocode: ret = sub0 + sub1

    Parameters
    ----------
    sub0: xarray.core.dataset.Dataset
        the primary subtable to be copied and used as the base for the returned value
    sub1: xarray.core.dataset.Dataset
        the subtable to append to sub0
    relational_ids_map: dictionary
        Mapped values to apply to sub1 prior to appending it to sub0.
        Keys are data_var or coordinate names.
        Values are dictionaries of mapping value, with "from" keys and "to" values.
        For example, if the source_id needs to be bumped up by 5 in sub1 (to match the
        earlier joining of the SOURCE subtable), then the map might be:
        {"source_id":{0:5, 1:6, 2:7}}

    Returns
    -------
    xarray.core.dataset.Dataset
        A new subtable, which the intention that it will be used to replace the
        current subtable in sub0.
    dictionary
        Dimensional coordinates in the sub1 subtable that got changed in order
        to be merged with the subtable in sub0. Example for the subtable FIELD:
        {"field_id":{0:5, 1:6, 2:7, 3:8, 4:9}}
    """
    # take care of "none" type parameters
    if (relational_ids_map == None):
        relational_ids_map = {}

    # check parameters
    if ('DATA' in list(sub0.data_vars)) or ('CORRECTED_DATA' in list(sub0.data_vars)) or \
       ('DATA' in list(sub1.data_vars)) or ('CORRECTED_DATA' in list(sub1.data_vars)):
        print("Warning: subtable should not be a visibility xds, but rather one of the 'global' tables. Found 'DATA' or 'CORRECTED_DATA' variable, which is usually only found in a visibility xds.")

    # get the dimensional coordinates
    # TODO what do we do about multidimensional coordinates?
    dim_coords0 = list(sub0.coords) # this returns the coordinate names as a list
    dim_coords1 = list(sub1.coords)
    dim_coords0 = list(filter(lambda n: n in sub0.dims, dim_coords0))
    dim_coords1 = list(filter(lambda n: n in sub1.dims, dim_coords1))
    if (dim_coords0 != dim_coords1):
        print("Warning: subtable dimensional coordinates do not match. Is it possible that sub0 and sub1 are from different converter versions?")
        print(f"sub0 dimensional coordinates: {dim_coords0}")
        print(f"sub1 dimensional coordinates: {dim_coords1}")

    # Get coordinates to bump in sub1.
    # These coordinate values will be used instead of the current coordinate
    # values from sub1 so that there aren't any collisions of coordinate
    # values in the joined subtable.
    coords_map = {}
    for coord_name in dim_coords0:
        # is there a collision in coordinate names?
        if (coord_name not in dim_coords1):
            continue
        # TODO issue a warning if the coordinates aren't integer values

        # get the map of new coordinate values
        coords_map[coord_name] = {}
        for coord_val in sub1[coord_name].values:
            # is there a collision in values?
            if (coord_val not in sub0[coord_name]):
                continue

            # map the current value to the next free id
            max_coord_val = 0
            if len(coords_map[coord_name]) > 0:
                max_coord_val = max(coords_map[coord_name].values())
            max_id = max([
                max(sub0.coords[coord_name].values),
                max(sub1.coords[coord_name].values),
                max_coord_val
            ])
            new_id = max_id + 1
            coords_map[coord_name][coord_val] = new_id

    # assign the new coordinate values
    for coord_name in coords_map.keys():
        # get a function that returns the mapped value if there is one, or otherwise returns the given value
        map_vals = coords_map[coord_name]
        map_func = lambda x: x if x not in map_vals else map_vals[x]

        # apply the new coordinate
        sub1 = _remap_coord(sub1, coord_name, map_func)

    # apply the relational_ids_map
    for var_name in relational_ids_map:
        # get a function that returns the mapped value if there is one, or otherwise returns the given value
        map_vals = relational_ids_map[var_name]
        map_func = lambda x: x if x not in map_vals else map_vals[x]
        # apply the mapping
        if var_name in sub1.coords:
            sub1 = _remap_coord(sub1, var_name, map_func)
        elif var_name in sub1.data_vars:
            sub1 = _remap_data_var(sub1, var_name, map_func)

    # Attempt to merge the subtables.
    # Throws a xarray.core.merge.MergeError if there are still dimension conflicts.
    ret = sub0.merge(sub1)

    return ret, coords_map