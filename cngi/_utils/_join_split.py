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
from typing import Tuple as type_Tuple

def _apply_coord_remap(xds : xr.Dataset, coord_name : str, map_func) -> xr.Dataset:
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

def _apply_data_var_remap(xds : xr.Dataset, var_name : str, map_func) -> xr.Dataset:
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

def get_subtable_matching_dimcoords(sub0: xr.Dataset, sub1 : xr.Dataset, matchtype="exact") -> dict:
    """!!!COMPUTES!!! Finds the dimensional coordinate values in sub0 and sub1 which represent the same thing.

    Extended Summary
    ----------------
    Limitations:
    Only works for primary dimensional coordinates (dimcoords that are first
    in the list of dimensions for dependent coords/data_vars).

    Parameters
    ----------
    sub0: xarray.Dataset
        mxds subtable to compare to sub1
    sub1: xarray.Dataset
        mxds subtable to compare to sub0
    matchtype: str
        How to match. Options are "none" or "exact".
        "none": return an empty dictionary (does not compute)
        "exact": all data_vars for the given dimensional coordinate must match. (computes)

    Returns
    -------
    dict
        The map of dimcoords that match between sub0 and sub1.
        Keys are the name of the coordinates. Values are dictionaries that map
        from the sub1 coordinate values to the sub0 coordinate values.
    """
    # check parameters
    assert(isinstance(sub0, xr.Dataset)), f"######### ERROR: subtable must be a dataset!"
    assert(isinstance(sub1, xr.Dataset)), f"######### ERROR: subtable must be a dataset!"

    # easy case
    if (matchtype != "exact"):
        return {}

    # get the dimensional coordinates
    # TODO what do we do about multidimensional coordinates?
    dim_coords0 = list(sub0.coords)  # this returns the coordinate names as a list
    dim_coords1 = list(sub1.coords)
    dim_coords0 = list(filter(lambda n: n in sub0.dims, dim_coords0))
    dim_coords1 = list(filter(lambda n: n in sub1.dims, dim_coords1))

    # Check for equality in dependent coordinates/data_vars.
    coords_and_vars = list(sub1.coords) + list(sub1.data_vars)
    ret = {}
    for dimcoord_name in dim_coords1:
        # don't worry about dimcoords that aren't in sub0
        if (dimcoord_name not in dim_coords0):
            continue

        # build the dictionary of vals to check
        vals_match = { }
        for dimcoord_val in sub1[dimcoord_name].values:
            vals_match[dimcoord_val] = True

        # Compare each dependent value along the axis to determine which
        # dimcoord values match between sub0 and sub1.
        for name in coords_and_vars:
            # don't compare the dimcoord to itself
            # don't worry about coords/data_vars not in sub0
            if (name == dimcoord_name):
                continue
            if (name not in sub0):
                continue

            # get dimensions of the this coordinate/variable
            # only continue if the dimcoord is the first dimension
            vals0 = sub0[name]
            vals1 = sub1[name]
            dims0 = list(vals0.dims)
            dims1 = list(vals1.dims)
            if (len(dims1) == 0) or (dims1[0] != dimcoord_name):
                continue
            assert(dims0[0] == dimcoord_name), f"######### ERROR: subtables structure mismatch! sub0.{name}.dims:{dims0}, sub1.{name}.dims:{dims1}!"

            # check for equality (should be delayed)
            for dimcoord_val in sub1[dimcoord_name].values:
                if not vals_match[dimcoord_val]:
                    continue # no point in checking
                vals_match[dimcoord_val] &= vals1[dimcoord_val].broadcast_equals(vals0[dimcoord_val])

        # add all matching dimcoord values to the returned match dictionary
        for dimcoord_val in sub1[dimcoord_name].values:
            if vals_match[dimcoord_val]:
                if dimcoord_name not in ret:
                    ret[dimcoord_name] = {}
                sub0_dimcoord_val = dimcoord_val # just compare like values for the moment
                ret[dimcoord_name][dimcoord_val] = sub0_dimcoord_val

    return ret

def _get_subtable_dimcoord_remap(sub0: xr.Dataset, sub1 : xr.Dataset, matching_dimcoords : dict=None, print_warnings=True) -> dict:
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
        represent the same thing. These override any values in the returned
        dict.
    print_warnings: bool
        True to print warnings. False otherwise.

    Returns
    -------
    dict
        The map of dimensional coordinate values to change in sub1.
        Keys are the name of the coordinates. Values are dictionaries that map
        from the current sub1 coord values to new suggested values.
    """
    # take care of "None" type parameters
    if (matching_dimcoords == None):
        matching_dimcoords = {}

    # check parameters
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
            # is there even a collision in values?
            if (coord_val not in sub0[coord_name]):
                continue

            # are these values already considered matching?
            if (coord_name in matching_dimcoords) and (coord_val in matching_dimcoords[coord_name]):
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

def append_xds_subtable(sub0 : xr.Dataset, sub1 : xr.Dataset, coords_vals_remap=None, matchtype="exact", relational_ids_map=None) -> type_Tuple[xr.Dataset, dict]:
    """!!!COMPUTES!!! Append the given subtable sub1 to a new subtable based on sub0. pseudocode: ret = sub0.append(sub1)

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
    coords_vals_remap: dict
        How to bump the coordinate values in sub1 so that they don't collide with
        sub0. If none, then this is filled with calls to get_subtable_matching_dimcoords(matchtype)
        and _get_subtable_dimcoord_remap().
    matchtype: str
        Used for the call to get_subtable_matching_dimcoords, if
        coords_vals_remap is None.
    relational_ids_map: dictionary
        Mapped values to apply to sub1 prior to appending it to sub0.
        Keys are data_var or coordinate names.
        Values are dictionaries of mapping value, with "from" keys and "to" values.
        For example, if the source_id needs to be bumped up by 5 in sub1
        (to match the earlier joining of the SOURCE subtable), then the map
        might be: {"source_id":{0:5, 1:6, 2:7}}

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
    if (coords_vals_remap == None):
        matching_dimcoords = get_subtable_matching_dimcoords(matchtype)
        coords_vals_remap = _get_subtable_dimcoord_remap(sub0, sub1, matching_dimcoords=matching_dimcoords)

    # assign the new coordinate values in sub1
    for coord_name in coords_vals_remap.keys():
        # skip unknown coordinates
        if not coord_name in sub1.coords:
            continue

        # get a function that returns the mapped value if there is one, or otherwise returns the given value
        map_vals = coords_vals_remap[coord_name]
        map_func = lambda x: x if x not in map_vals else map_vals[x]

        # apply the new coordinate
        sub1 = _apply_coord_remap(sub1, coord_name, map_func)

    # apply the relational_ids_map to sub1
    for var_name in relational_ids_map:
        # get a function that returns the mapped value if there is one, or otherwise returns the given value
        map_vals = relational_ids_map[var_name]
        map_func = lambda x: x if x not in map_vals else map_vals[x]
        # apply the mapping
        if var_name in sub1.coords:
            sub1 = _apply_coord_remap(sub1, var_name, map_func)
        elif var_name in sub1.data_vars:
            sub1 = _apply_data_var_remap(sub1, var_name, map_func)

    # Attempt to merge the subtables.
    # Throws a xarray.core.merge.MergeError if there are still dimension conflicts.
    ret = sub0.merge(sub1, compat='broadcast_equals', join='outer')

    return ret, coords_vals_remap

def append_mxds_subtables(mxds0 : xr.Dataset, mxds1 : xr.Dataset, matchtype="exact") -> xr.Dataset:
    """!!!COMPUTES!!! Append the subtables of dataset mxds1 to the dataset mxds0 and return the new dataset.

    Extended Summary
    ----------------
    pseudocode: ret = mxds0.append(mxds1.coords).append(mxds1.attrs.sel(!"xds*"))

    This function makes heavy use of append_xds_subtable to append subtables
    from mxds1 to mxds0 in a sort of round-robin style. The resulting
    subtables are merged into a new mxds to be returned.

    The new mxds maintains all dims and coords of mxds0, with possibly expanded
    values for the dimensional coordinates.

    If a subtable or coordinate appears in mxds1 that isn't in mxds0, it is
    included in the returned result.

    Parameters
    ----------
    mxds0: xarray.Dataset
        The dataset to copy and append to.
    mxds1: xarray.Dataset
        The dataset to source new subtables from to append to the subtables in mxds0.
    matchtype: str
        How to determine dimension coordinate equality. Dimcoord value equality
        is then used to determine which mxds1 coordinate and data_var values
        need to be remapped to new coordinates and merged, and which values can
        be safely ignored in the merge.
        See get_subtable_matching_dimcoords() for a description of possible
        values.

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
        sub0 = mxds0.attrs[subtable_name]
        sub1 = subtables[subtable_name]['table'] # type: xr.Dataset
        if subtables[subtable_name]['in_mxds0']:
            matching_dimcoords = get_subtable_matching_dimcoords(sub0, sub1, matchtype=matchtype)
            coords_vals_remap = _get_subtable_dimcoord_remap(sub0, sub1, matching_dimcoords)
            for coord_name in coords_vals_remap:
                assert(coord_name not in ids_map), f"######### ERROR: subtables can't share the same dimensional coordinates! Offending subtable/coordinate: {subtable_name}/{coord_name}"
                ids_map[coord_name] = coords_vals_remap[coord_name]

    # Build out the subtables by appending those in mxds1 to the ones in mxds0.
    # Pass the dimcoord remapping dictionary in to update dimcoord values in the mxds1 subtables.
    # Pass ids_map through to keep all the reference indexes consistent in the remapped mxds1 subtables.
    # TODO check that new ref ids aren't out-of-bounds
    # TODO unit test for ref id updates
    attrs = mxds0.attrs.copy()
    for subtable_name in subtables:
        if subtables[subtable_name]['in_mxds0']:
            # this is a shared subtable, append it
            sub0 = mxds0.attrs[subtable_name]
            sub1 = subtables[subtable_name]['table']
            new_subtable = append_xds_subtable(sub0, sub1, coords_vals_remap=ids_map, relational_ids_map=ids_map)
            attrs[subtable_name] = new_subtable
        else:
            # include all subtables in mxds1 that aren't in mxds0
            attrs[subtable_name] = subtables[subtable_name]['table']

    # merge mxds0 and mxds1
    ret = xr.Dataset(data_vars=mxds0.data_vars, coords=mxds0.data_vars, attrs=attrs)
    return ret
