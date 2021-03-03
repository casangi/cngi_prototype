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
    """Apply the map_func to the values in the given coordinate"""
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
    """Apply the map_func to the values in the given data_var"""
    import numpy as np
    def mb(array):
        vals = array.values
        if (len(vals) > 0):
            vals = np.vectorize(map_func)(vals)
        return xr.DataArray(data=vals, coords=array.coords, dims=array.dims, name=array.name, attrs=array.attrs)

    assert (isinstance(xds[var_name], xr.DataArray)), f"######### ERROR: trying to remap the data variable {var_name} which is a {type(xds[var_name])} but a {xr.DataArray} was expected!"
    var_val = xds[var_name].map_blocks(mb)
    return xds.assign({var_name: var_val})

def _get_subtable_dimcoord_bump(sub0: xr.Dataset, sub1 : xr.Dataset, matching_dimcoords : dict=None, print_warnings=True) -> dict:
    """Finds the dimensional coordinate values that collide between sub0 and sub1, and maps the dimcoord values in sub1 to no longer collide.

    Parameters
    ----------
    sub0: xarray.Dataset
        mxds subtable to compare the dimcoord values of sub1 against.
    sub1: xarray.Dataset
        mxds subtable to find colliding dimcoord values in.
    matching_dimcoords: dict
        Known dimensional coordinate values that represent the same thing
        between sub0 and sub1. Keys are the dimcoord name. Values are a map
        from the dimcoord values in sub1 to the dimcoord values in sub0 that
        represent the same thing. These are used as the initial values in the
        returned dict.
    print_warnings: bool
        True to print warnings. False otherwise.

    Returns
    -------
    dict
        The map of dimensional coordinate values to change in sub1.
        Keys are the name of the coordinates. Values are dictionaries that map
        from the current sub1 coord values to new suggested values.
    """

    # check parameters
    if (matching_dimcoords == None):
        matching_dimcoords = {}
    assert(isinstance(sub0, xr.Dataset)), f"######### ERROR: subtable must be a dataset!"
    assert(isinstance(sub1, xr.Dataset)), f"######### ERROR: subtable must be a dataset!"

    # get the dimensional coordinates
    # TODO what do we do about multidimensional coordinates?
    dim_coords0 = list(sub0.coords)  # this returns the coordinate names as a list
    dim_coords1 = list(sub1.coords)
    dim_coords0 = list(filter(lambda n: n in sub0.dims, dim_coords0))
    dim_coords1 = list(filter(lambda n: n in sub1.dims, dim_coords1))
    if print_warnings and (dim_coords0 != dim_coords1):
        print("Warning: subtable dimensional coordinates do not match. Is it possible that sub0 and sub1 are from different converter versions?")
        print(f"sub0 dimensional coordinates: {dim_coords0}")
        print(f"sub1 dimensional coordinates: {dim_coords1}")

    # Get coordinates to bump in sub1.
    # These coordinate values will be used instead of the current coordinate
    # values from sub1 so that there aren't any collisions of coordinate
    # values in the joined subtable.
    coords_map = { }
    for coord_name in dim_coords0:
        # is there a collision in coordinate names?
        if (coord_name not in dim_coords1):
            continue
        # TODO issue a warning if the coordinates aren't integer values

        # get the map of new coordinate values
        coords_map[coord_name] = { }
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
            new_id = max_id+1
            coords_map[coord_name][coord_val] = new_id

    return coords_map

def append_xds_subtable(sub0 : xr.Dataset, sub1 : xr.Dataset, relational_ids_map=None):
    """ Append the given subtable sub1 to a new subtable based on sub0. pseudocode: ret = sub0.append(sub1)

    Extended Summary
    ----------------
    Append sub1 to sub0 and return the result. Before appending sub1:
    1. calculate new dimensional coordinate values for sub1 so that sub1 and sub0 DCs won't conflict
    2. apply DC changes to sub1_copy
    3. apply the maps in relation_ids_map to the coordinates/data_vars in sub1_copy
    Finally, sub0 and sub1_copy are merged and returned.

    Parameters
    ----------
    sub0: xarray.Dataset
        the primary subtable to be copied and used as the base for the returned value
    sub1: xarray.Dataset
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
    xarray.Dataset
        A new subtable, which the intention that it will be used to replace the
        current subtable in sub0.
    dict
        Dimensional coordinates in the sub1 subtable that got changed in order
        to be merged with the subtable in sub0. Example for the subtable FIELD:
        {"field_id":{0:5, 1:6, 2:7, 3:8, 4:9}}
    """
    # take care of "none" type parameters
    if (relational_ids_map == None):
        relational_ids_map = {}

    # check parameters
    assert(isinstance(sub0, xr.Dataset)), f"######### ERROR: subtable must be a dataset!"
    assert(isinstance(sub1, xr.Dataset)), f"######### ERROR: subtable must be a dataset!"
    if ('DATA' in list(sub0.data_vars)) or ('CORRECTED_DATA' in list(sub0.data_vars)) or \
       ('DATA' in list(sub1.data_vars)) or ('CORRECTED_DATA' in list(sub1.data_vars)):
        print("Warning: subtable should not be a visibility xds, but rather one of the 'global' tables. Found 'DATA' or 'CORRECTED_DATA' variable, which is usually only found in a visibility xds.")

    # get the dimension coordinates to be bumped
    coords_map = _get_subtable_dimcoord_bump(sub0, sub1)

    # assign the new coordinate values in sub1
    for coord_name in coords_map.keys():
        # get a function that returns the mapped value if there is one, or otherwise returns the given value
        map_vals = coords_map[coord_name]
        map_func = lambda x: x if x not in map_vals else map_vals[x]

        # apply the new coordinate
        sub1 = _remap_coord(sub1, coord_name, map_func)

    # apply the relational_ids_map to sub1
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

def append_mxds(mxds0 : xr.Dataset, mxds1 : xr.Dataset):
    """Append the dataset mxds1 to the dataset mxds0 and return the new dataset. pseudocode: ret = mxds0.drop_attrs(xds*).append(mxds1)

    Extended Summary
    ----------------
    This function makes heavy use of append_xds_subtable to append subtables
    from mxds1 to mxds0 in a sort of round-robin style. The resulting
    subtables are merged into a new mxds to be returned.
    The new mxds maintains all dims and coords of mxds0, but none of the
    visibilities "xds*".
    If a subtable appears in mxds1 that isn't in mxds0, it is included in the
    returned result.

    Parameters
    ----------
    mxds0: xarray.Dataset
        The dataset to copy and append to.
    mxds1: xarray.Dataset
        The dataset to source new subtables from to append to the subtables in mxds0.

    Returns
    -------
    xarray.Dataset
        mxds0, plut the the appended subtables from mxds1.
    """
    # get the list of subtables to append to mxds0 from mxds1
    subtables = {}
    for subtable_name in list(mxds1.attrs):
        # don't include visibility tables
        # don't include anything that isn't a subtable
        if ("xds" in subtable_name) or (not isinstance(mxds1.attrs[subtable_name], xr.Dataset)):
            continue
        # record the subtable
        subtables[subtable_name] = {
            "table": mxds1.attrs[subtable_name],
            "in_mxds0": (subtable_name in mxds0.attrs)
        }

    # Get the coordinate remapping for coordinate relationships between mxds1
    # subtables, and check that there are no dimensional coordinates conflicts
    # between any of the subtables in mxds1.
    ids_map = {}
    for subtable_name in subtables:
        print(subtable_name)
        sub = subtables[subtable_name]['table'] # type: xr.Dataset
        if subtables[subtable_name]['in_mxds0']:
            print("in mxds0")
            dimcoord_changes = _get_subtable_dimcoord_bump(mxds0.attrs[subtable_name], sub, print_warnings=False)
            print(f"dimcoord_changes: {dimcoord_changes}")
            for coord_name in dimcoord_changes:
                assert(coord_name not in ids_map), f"######### ERROR: subtables can't share the same dimensional coordinates! Offending subtable/coordinate: {subtable_name}/{coord_name}"
                ids_map[coord_name] = dimcoord_changes[coord_name]

    return ids_map
    # # Build out the subtables by appending those in mxds1 to the ones in mxds0.
    # # Pass coord_bumps through to keep all the reference indexes consistent.
    # # TODO check that new ref ids aren't out-of-bounds
    # # TODO unit test for ref id updates
    # attrs = {}
    # for attr_name in mxds0.attrs:
    #     if attr_name not in subtables:
    #         # not a subtable, nothing to do but copy it
    #         attrs[attr_name] = mxds0.attrs[attr_name]
    #     else:
    #         # this is a shared subtable, append it
    #         new_subtable = append_xds_subtable(sub0=mxds0.attrs[attr_name], sub1=subtables[attr_name], relational_ids_map=ids_map)
    #         attrs[attr_name] = new_subtable
    # # append any subtables that aren't in mxds0
    # for subtable_name in subtables:
    #     if not subtables[subtable_name].in_mxds0:
    #         attrs[subtable_name] = subtables[subtable_name].table
    #
    # # merge mxds0 and mxds1
    # ret = xr.Dataset(data_vars=mxds0.data_vars, coords=mxds0.data_vars, attrs=attrs)
    # return ret