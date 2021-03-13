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
import typing

def apply_coord_remap(xds : xr.Dataset, coord_name : str, map_func) -> xr.Dataset:
    """Apply the map_func to the values in the given coordinate"""
    import numpy as np

    # get the new values
    old_vals = xds[coord_name].values
    new_vals = np.array([map_func(x) for x in old_vals], old_vals.dtype)

    # assign the coordinate
    if coord_name in xds.dims:
        # assign as dimensional coordinate
        ret = xds.drop_vars([ coord_name ])
        ret = ret.assign_coords({ coord_name: new_vals })
    else:
        # assign as non-dimensional coordinate
        coord_val = xr.DataArray(new_vals, dims=xds[coord_name].dims)
        ret = xds.drop_vars([ coord_name ])
        ret = ret.assign({ coord_name: coord_val }) # creates coord_name as a data_var
        ret = ret.set_coords([ coord_name ]) # promote the new data_var "coord_name" to be a coordinate
    return ret

def apply_data_var_remap(xds : xr.Dataset, var_name : str, map_func) -> xr.Dataset:
    """Apply the map_func to the values in the given data_var"""
    import numpy as np
    def mb(array):
        vals = array.values
        newvals = np.ndarray(vals.shape, vals.dtype)
        if len(vals) > 0:
            newvals = np.vectorize(map_func, [vals.dtype])(vals)
        return xr.DataArray(data=newvals, coords=array.coords, dims=array.dims, name=array.name, attrs=array.attrs)

    assert (isinstance(xds[var_name], xr.DataArray)), f"######### ERROR: trying to remap the data variable {var_name} which is a {type(xds[var_name])} but a {xr.DataArray} was expected!"
    var_val = xds[var_name].map_blocks(mb)
    return xds.assign({var_name: var_val})

def _get_subtable_dimcoords_or_primcoords(sub : xr.Dataset, subtable_name : str) -> typing.List[str]:
    """Get the dimension coordinate and primary key coordinate names from the given subtable"""
    import numpy as np
    from cngi._utils._mxds_ops import get_subtable_primary_key_names
    # TODO what do we do about multidimensional coordinates?
    dim_coords = list(sub.coords)  # this returns the coordinate names as a list
    dim_coords = list(filter(lambda n: n in sub.dims, dim_coords))
    primary_key_names = get_subtable_primary_key_names(sub, subtable_name)
    return list(np.unique(dim_coords + primary_key_names))

def _get_subtable_matching_dimcoords(sub0: xr.Dataset, sub1 : xr.Dataset, subtable_name : str, matchtype="exact") -> dict:
    """!!!COMPUTES!!! Finds the dimensional/primary coordinate values in sub0 and sub1 which represent the same thing.

    Extended Summary
    ----------------
    Limitations:
    Only works for 0-axis coordinates (those that are first in the list of
    dimensions for dependent coords/data_vars).

    Parameters
    ----------
    sub0: xarray.Dataset
        mxds subtable to compare to sub1
    sub1: xarray.Dataset
        mxds subtable to compare to sub0
    subtable_name: str
        Name of the subtables sub0 and sub1. Example "FIELD".
    matchtype: str
        How to match. Options are "none" or "exact".
        "none": return an empty dictionary (does not compute)
        "exact": all data_vars for the given coordinate must match. (computes)

    Returns
    -------
    dict
        The map of dimensional/primary coordinate that match between sub0 and sub1.
        Keys are the name of the coordinates. Values are dictionaries that map
        from the sub1 coordinate values to the sub0 coordinate values.
    """
    import warnings
    # check parameters
    assert(isinstance(sub0, xr.Dataset)), f"######### ERROR: subtable {subtable_name} must be a dataset!"
    assert(isinstance(sub1, xr.Dataset)), f"######### ERROR: subtable {subtable_name} must be a dataset!"

    # easy case
    if (matchtype != "exact"):
        return {}

    # get the dimensional/primary coordinates
    check_coords0 = _get_subtable_dimcoords_or_primcoords(sub0, subtable_name)
    check_coords1 = _get_subtable_dimcoords_or_primcoords(sub1, subtable_name)

    # Check for equality in dependent coordinates/data_vars.
    coords_and_vars = list(sub1.coords) + list(sub1.data_vars)
    ret = {}
    for coord_name in check_coords1:
        # don't worry about dimcoords that aren't in sub0
        if coord_name not in check_coords0:
            continue

        # build the dictionary of vals to check
        vals_match = { }
        for coord_val in sub1[coord_name].values:
            vals_match[coord_val] = True

        # compare values based on the 0-axis dimension
        dimname = coord_name
        if coord_name not in sub1.dims:
            dimname = sub1[coord_name].dims[0]

        # Compare each dependent value along the axis to determine which
        # dimcoord values match between sub0 and sub1.
        for name in coords_and_vars:
            # don't compare the dimcoord to itself
            # don't worry about coords/data_vars not in sub0
            if name == coord_name:
                continue
            if name not in sub0:
                continue

            # get dimensions of the this coordinate/variable
            # stop if the dimcoord is not the first dimension
            vals0 = sub0[name]
            vals1 = sub1[name]
            dims0 = list(vals0.dims)
            dims1 = list(vals1.dims)
            if (len(dims1) == 0) or (dims1[0] != dimname):
                continue
            assert(dims0[0] == dimname), f"######### ERROR: subtables structure mismatch! {subtable_name}0.{name}.dims:{dims0}, {subtable_name}1.{name}.dims:{dims1}!"

            # check for equality
            for coord_val in sub1[coord_name].values:
                if not vals_match[coord_val]:
                    continue # no point in checking
                # Do any necessary computation here to hopefully catch errors
                # without getting into any messy broadcast_equals code.
                vals0[coord_val].compute()
                vals1[coord_val].compute()
                # not a lot that we can do about the way dask does comparisons
                # https://stackoverflow.com/questions/40659212/futurewarning-elementwise-comparison-failed-returning-scalar-but-in-the-futur
                with warnings.catch_warnings():
                    warnings.simplefilter(action='ignore', category=FutureWarning)
                    vals_match[coord_val] &= vals1[coord_val].broadcast_equals(vals0[coord_val])

        # add all matching dimcoord values to the returned match dictionary
        for coord_val in sub1[coord_name].values:
            if vals_match[coord_val]:
                if coord_name not in ret:
                    ret[coord_name] = {}
                sub0_dimcoord_val = coord_val # just compare like values for the moment
                ret[coord_name][coord_val] = sub0_dimcoord_val

    return ret

def _get_subtable_dimcoord_remap(sub0: xr.Dataset, sub1 : xr.Dataset, subtable_name : str, matching_dimcoords : dict=None, existing_map : dict=None, print_warnings=True) -> typing.Dict[str, dict]:
    """Finds the dimensional/primary coordinate values that collide between sub0 and sub1, and maps the values in sub1 to no longer collide with the values in sub0.

    Parameters
    ----------
    sub0: xarray.Dataset
        mxds subtable to compare the dimcoord values of sub1 against.
    sub1: xarray.Dataset
        mxds subtable to find colliding dimcoord values in.
    subtable_name: str
        Name of the subtables sub0 and sub1. Example "FIELD".
    matching_dimcoords: dict
        Known dimensional/primary coordinate values that represent the same
        thing between sub0 and sub1. Keys are the coordinate name. Values are a
        map from the coordinate values in sub1 to the coordinate values in sub0
        that represent the same thing. These override any values in the
        returned dict.
    existing_map: dict
        Dimensional/primary coordinates values that already have a mapping,
        most likely from a previous evaluation of this function. Keys are the
        coordinate name. Values are a map from the key values in sub1 to the
        new key values that sub1 is going to be changed to use.
    print_warnings: bool
        True to print warnings. False otherwise.

    Returns
    -------
    dict
        The map of dimensional/primary coordinate values to change in sub1.
        Keys are the name of the coordinates. Values are dictionaries that map
        from the current sub1 coord values to new suggested values.
    """
    from cngi._utils._mxds_ops import get_subtable_primary_key_names
    import numpy as np

    # take care of "None" type parameters
    if (matching_dimcoords == None):
        matching_dimcoords = {}
    if (existing_map == None):
        existing_map = {}

    # check parameters
    assert(isinstance(sub0, xr.Dataset)), f"######### ERROR: subtable {subtable_name} must be a dataset!"
    assert(isinstance(sub1, xr.Dataset)), f"######### ERROR: subtable {subtable_name} must be a dataset!"

    # get the dimensional/primary coordinates
    check_coords0 = _get_subtable_dimcoords_or_primcoords(sub0, subtable_name)
    check_coords1 = _get_subtable_dimcoords_or_primcoords(sub1, subtable_name)
    if print_warnings and (check_coords0 != check_coords1):
        print("Warning: subtable primary keys do not match. Is it possible that sub0 and sub1 are from different converter versions?")
        print(f"sub0 primary keys: {check_coords0}")
        print(f"sub1 primary keys: {check_coords1}")

    # Get coordinates to bump in sub1.
    # These coordinate values will be used instead of the current coordinate
    # values from sub1 so that there aren't any collisions of coordinate
    # values in the joined subtable.
    coords_map = { }
    for coord_name in check_coords0:
        # is there a collision in coordinate names?
        if (coord_name not in check_coords1):
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
            # do these values already have a mapping?
            if (coord_name in existing_map) and (coord_val in existing_map[coord_name].values()):
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
    
def _remap_subtable_coords_and_vals(sub1 : xr.Dataset, subtable_name : str, coords_vals_remap : dict, relational_ids_map : dict, update_refs_only=False) -> xr.Dataset:
    """!!!COMPUTES!!! Update subtable sub1 with the given coords_vals_remap and relation_ids_map. pseudocode: ret = sub1.update(coords_vals_remap, relational_ids_map)

    Extended Summary
    ----------------
    Copy sub1.
    Apply the maps in coords_vals_remap to the coordinates/data_vars in sub1_copy.
    Apply the maps in relation_ids_map to the coordinates/data_vars in sub1_copy.

    Parameters
    ----------
    sub1: xarray.Dataset
        the subtable to be changed
    subtable_name: str
        Name of the subtable sub1. Example "FIELD".
    coords_vals_remap: dict
        How to bump the coordinate values in sub1 so that they don't collide
        with sub0 (see append_xds_subtable).
    relational_ids_map: dict
        Mapped values to apply to sub1.
        Keys are data_var or coordinate names.
        Values are dictionaries of mapping value, with "from" keys and "to" values.
        For example, if the source_id needs to be bumped up by 5 in sub1
        (to match the earlier joining of the SOURCE subtable), then the map
        might be: {"source_id":{0:5, 1:6, 2:7}}
    update_refs_only: bool
        If true, then dimensional coordinates and primary keys are not updated.
        Instead, only values that are in the coords_val_remap or
        relation_ids_map that don't meet this criteria are updated.

    Returns
    -------
    xarray.Dataset
        A new subtable, which the intention that it will be appended to sub0 in
        append_xds_subtable(...).
    """
    # check parameters
    assert(isinstance(sub1, xr.Dataset)), f"######### ERROR: subtable {subtable_name} must be a dataset!"

    # get the list of non-reference coordinates
    nonref_coord_names = _get_subtable_dimcoords_or_primcoords(sub1, subtable_name)

    # assign the new coordinate values in sub1
    for coord_name in coords_vals_remap.keys():
        # skip unknown coordinates
        if not coord_name in sub1.coords:
            continue

        # skip non-reference coordinates
        if update_refs_only and coord_name in nonref_coord_names:
            continue

        # get a function that returns the mapped value if there is one, or otherwise returns the given value
        map_vals = coords_vals_remap[coord_name]
        map_func = lambda x: x if x not in map_vals else map_vals[x]

        # apply the new coordinate
        sub1 = apply_coord_remap(sub1, coord_name, map_func)

    # apply the relational_ids_map to sub1
    for var_name in relational_ids_map:
        # skip non-reference coordinates
        if update_refs_only and var_name in nonref_coord_names:
            continue

        # get a function that returns the mapped value if there is one, or otherwise returns the given value
        map_vals = relational_ids_map[var_name]
        map_func = lambda x: x if x not in map_vals else map_vals[x]

        # apply the mapping
        if var_name in sub1.coords:
            sub1 = apply_coord_remap(sub1, var_name, map_func)
        elif var_name in sub1.data_vars:
            sub1 = apply_data_var_remap(sub1, var_name, map_func)

    return sub1

def _build_mxds_coords(mxds : xr.Dataset, subtables : typing.Dict[str, xr.Dataset]):
    from cngi._utils._mxds_ops import get_subtable_primary_key_names

    # make a copy of the current coordinates, as a dictionary so that it can be easily updated
    coords = dict(mxds.coords).copy()

    # build new coordinates based on the primary keys in each of the subtables
    for sn in subtables:
        sub = subtables[sn]

        # update the global coordinate with the matching primary key name "coord_*_name"
        primary_key_names = get_subtable_primary_key_names(sub, subtable_name=sn)
        for pkname in primary_key_names:

            # find the coord_name, example "antenna_ids" and "antennas"
            coord_id = sn.lower().replace("spectral_window", "spw")+"_ids"
            coord_name = sn.lower().replace("spectral_window", "spw").replace("_id", "")+"s"

            # update coordinate ids
            if coord_id in coords:
                coords[coord_id] = (coord_id, sub.coords[pkname].values)
            if (coord_name in coords) and ("NAME" in sub.data_vars):
                coords[coord_name] = (coord_id, sub.data_vars["NAME"].values)

    return coords

def append_xds_subtable(sub0 : xr.Dataset, sub1 : xr.Dataset) -> xr.Dataset:
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

    Returns
    -------
    xarray.Dataset
        A new subtable, which the intention that it will be used to replace the
        current subtable in sub0.
    """
    # check parameters
    assert(isinstance(sub0, xr.Dataset)), f"######### ERROR: subtable {subtable_name} must be a dataset!"
    assert(isinstance(sub1, xr.Dataset)), f"######### ERROR: subtable {subtable_name} must be a dataset!"
    if ('DATA' in list(sub0.data_vars)) or ('CORRECTED_DATA' in list(sub0.data_vars)) or \
       ('DATA' in list(sub1.data_vars)) or ('CORRECTED_DATA' in list(sub1.data_vars)):
        print("Warning: subtable should not be a visibility xds, but rather one of the 'global' tables. Found 'DATA' or 'CORRECTED_DATA' variable, which is usually only found in a visibility xds.")

    # Attempt to merge the subtables.
    # Throws a xarray.core.merge.MergeError if there are still dimension conflicts.
    compat='no_conflicts' # only check for equality in non-nan values
    join='outer' # retain all coordinate values
    ret = sub0.merge(sub1, compat=compat, join=join)

    return ret

def append_mxds_subtables(mxds0 : xr.Dataset, mxds1 : xr.Dataset, matchtype="exact") -> typing.Tuple[xr.Dataset, typing.Dict[str,dict]]:
    """!!!COMPUTES!!! Append the subtables of dataset mxds1 to the dataset mxds0 and return the new dataset.

    Extended Summary
    ----------------
    pseudocode: ret = mxds0.append(mxds1.coords).append(mxds1.attrs.sel(!"xds*"))

    This function makes heavy use of _remap_subtable_coords_and_vals to update
    references from mxds1 to mxds0 in a sort of round-robin style. The
    resulting subtables are merged into a new mxds to be returned.

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
        See _get_subtable_matching_dimcoords() for a description of possible
        values.

    Returns
    -------
    xarray.Dataset
        mxds0, plut the the appended subtables from mxds1.
    dict
        Map of which dimensional coordinates/primary keys were changed and
        their changed values. Keys are the names of the dimcoords/primekeys.
        Values are the mapping from old value in mxds1 to the new values in
        the returned dataset.
    """
    import copy
    from cngi._utils._mxds_ops import get_subtables, get_subtable_primary_key_names, check_mxds_subtable_ref_ids, assign_dimensions_for_primary_coordinates

    # make a copy of the subtables, as a dictionary so that subtables can be easily updated
    attrs = mxds0.attrs.copy() # type: typing.Dict[str, xr.Dataset]

    # get the list of subtables to append to mxds0 from mxds1
    subs    = {} # type: typing.Dict[str, xr.Dataset]
    sub_in0 = {} # type: typing.Dict[str, bool]
    for sn in get_subtables(mxds1):
        subs[sn]    = mxds1.attrs[sn]
        sub_in0[sn] = (sn in attrs)

    # Assign dimension coordinates for primary keys that aren't already dimension coordinates.
    # We do this so that there aren't going to be collisions when we try to merge mxds1 into mxds0.
    # Use the same name as the primary key.
    for sn in subs:
        if sub_in0[sn]:
            # this is a shared subtable, add dimension coordinates
            attrs[sn], subs[sn] = assign_dimensions_for_primary_coordinates(sub0=attrs[sn], sub1=subs[sn], subtable_name=sn)
        else:
            # not necessary for tables that only appear in mxds1
            pass

    # Get the keys to update and update them in the references.
    # The reason for updating references first is that these changes may cascade changes in the keys.
    # Once the key values have stabilized, then update the keys.
    ids_map = {} # type: typing.Dict[str, dict]
    keys_changed = True
    fresh_subs = subs.copy()
    while keys_changed:
        # Get the coordinate remapping for coordinate relationships between mxds1
        # subtables, and check that there are no dimensional coordinates conflicts
        # between any of the subtables in mxds1.
        new_ids_map = copy.deepcopy(ids_map)
        for sn in subs:
            sub0 = attrs[sn]
            sub1 = subs[sn]
            if sub_in0[sn]:
                matching_dimcoords = _get_subtable_matching_dimcoords(sub0, sub1, subtable_name=sn, matchtype=matchtype)
                coords_vals_remap = _get_subtable_dimcoord_remap(sub0, sub1, subtable_name=sn, matching_dimcoords=matching_dimcoords, existing_map=new_ids_map)
                for coord_name in coords_vals_remap:
                    # assert(coord_name not in ids_map), f"######### ERROR: subtables can't share the same dimensional coordinates! Offending subtable/coordinate: {subtable_name}/{coord_name}"
                    new_ids_map[coord_name] = coords_vals_remap[coord_name]
        keys_changed = (new_ids_map != ids_map)
        ids_map = new_ids_map

        # Pass the dimcoord remapping dictionary in to update dimcoord values in the mxds1 subtables.
        # Pass ids_map through to keep all the reference indexes consistent in the remapped mxds1 subtables.
        for sn in subs:
            # Update the references or keys of this subtable.
            if sub_in0[sn]:
                # For each loop, we want to start over with a fresh copy of sub1 because of complicated nonsense
                #   (the reference values could have changed, causing keys to change, causing reference values
                #    to change in some places but not in others).
                sub1 = fresh_subs[sn]
                new_subtable = _remap_subtable_coords_and_vals(sub1, subtable_name=sn, coords_vals_remap=ids_map, relational_ids_map=ids_map, update_refs_only=keys_changed)
                subs[sn] = new_subtable # use this new value for computing the next loop (above), or for appending (below)

    # Build out the new subtables by appending those in mxds1 to the ones in mxds0.
    # TODO unit test for ref id updates
    for sn in subs:
        if sub_in0[sn]:
            # this is a shared subtable, append it to the subtable from mxds0
            try:
                attrs[sn] = append_xds_subtable(sub0=attrs[sn], sub1=subs[sn])
            except xr.MergeError as e:
                print(f"Error in appending subtable {sn} in {append_mxds_subtables.__name__}")
                raise
        else:
            # include all subtables in mxds1 that aren't in mxds0 (copy sub1 to the missing sub0)
            attrs[sn] = subs[sn]

    # update the global coordinates to reflect the new coordinate values in the subtables
    coords = _build_mxds_coords(mxds0, attrs)

    # merge mxds0 and mxds1
    ret = xr.Dataset(data_vars=mxds0.data_vars, coords=coords, attrs=attrs)
    check_mxds_subtable_ref_ids(ret)

    return ret, ids_map

def gen_keyname_variants(keyname : str) -> typing.List[str]:
    """The dimensional coordinate names and primary coordinate names from the
    mxds subtables are mangled in their usage in the main xds visibility tables.
    This function provides all possible mangled versions that can be matched
    against for doing comparisons.

    Example keynames: antenna_id, beam_id, feed_id, spectral_window_id, ns_ws_station_id
    Example variants: "pol_id", "spw_id", "ANTENNA1", "ANTENNA2", "ARRAY_ID", "FEED1"
    """
    knvariants = []

    knl = keyname.lower()
    for kn1 in [knl, knl.replace("spectral_window", "spw")]:
        for kn2 in [kn1, kn1.replace("_id", "")]:
            for kn3 in [kn2, kn2.upper(), kn2+'s', kn2.upper()+'S']:
                knvariants.append(kn3)
            for kn3 in [kn2+'1', kn2+'2', kn2.upper()+'1', kn2.upper()+'2']:
                knvariants.append(kn3)

    return knvariants

def extract_xds_with_subtables(mxds : xr.Dataset, xds_names : typing.Union[str, typing.List[str]]) -> xr.Dataset:
    """Pull the xds visibilites out with the mxds, preserving only that information in the subtables that is related to the given xds.

    Parameters
    ----------
    mxds: xarray.Dataset
        The multi-xds dataset to pull data out of.
    xds_name: str or list
        Name(s) of the visibilities dataset. Should be of the form "xds*"

    Returns
    -------
    xarray.Dataset
        A new mxds, which includes just the xds_name visibility Dataset and the
        related information from the subtables.
    """
    import numpy as np
    from cngi._utils._mxds_ops import get_subtables, get_subtable_primary_key_names, check_mxds_subtable_ref_ids, assign_dimensions_for_primary_coordinates

    # get a list of the main tables
    main_tables = [] # type: typing.List[xr.Dataset]
    if isinstance(xds_names, str):
        xds_names = [xds_names]
    for xds_name in xds_names:
        assert ("xds" in xds_name), f"######### ERROR: xds_name must reference a main table! Name should contain \"xds\" but is instead {xds_name}!"
        assert (xds_name in mxds.attrs), f"######### ERROR: main table {xds_name} does not appear in the mxds list of attrs!"
        main = mxds.attrs[xds_name] # type: xr.Dataset
        assert (isinstance(main, xr.Dataset)), f"######### ERROR: xds visibilities table must be a Dataset but is instead a {type(main)}!"
        main_tables.append(main)

    # make a copy of the subtables, as a dictionary so that it can be easily updated
    # exclude all xds visibilities other than the desired main tables
    attrs = {} # type: typing.Dict[str, xr.Dataset]
    for sn in mxds.attrs: # sn = subtable_name
        if ("xds" not in sn) or (sn in xds_names):
            attrs[sn] = mxds.attrs[sn]

    # get the list of subtables, and the list of key coordinates used for indexing those subtables
    # example subtables: ANTENNA, ASDM_ANTENNA, FEED, WEATHER
    # example key coordinates: antenna_id, beam_id, feed_id, spectral_window_id -> spw_id, ns_ws_station_id
    subnames = get_subtables(mxds)
    sub_keynames = {}
    keynames = [] # type: typing.List[str]
    for sn in subnames:
        sub_kns = _get_subtable_dimcoords_or_primcoords(attrs[sn], sn)
        keynames += sub_kns
        sub_keynames[sn] = sub_kns
    keynames = list(np.unique(keynames))

    # get the list of key values to keep, based off of the main tables
    used_keyvals = {}
    used_knvariants = []
    for kn in keynames:
        # get a list of variant keynames that could be used in the main tables
        # example variants: "pol_id", "spw_id", "ANTENNA1", "ANTENNA2", "ARRAY_ID", "FEED1"
        knvariants = gen_keyname_variants(kn)

        # find the used values
        used = []
        for main in main_tables:
            for knvariant in knvariants:
                if (knvariant not in main.coords) and (knvariant not in main.data_vars):
                    continue
                used_knvariants.append(knvariant)
                vals = np.unique(main[knvariant].values) # get unique values along each dimension
                tmpused = np.unique(vals.flatten()) # flatten to a single dimension and get unique values along that single dimension
                tmpused = list(filter(lambda x: not np.isnan(x), tmpused))
                used += tmpused
        if len(used) == 0:
            used_keyvals[kn] = None
        else:
            used_keyvals[kn] = np.unique(used)
    # print(keynames)
    # print(np.unique(used_knvariants))

    # build a new set of subtables with trimmed values
    alldrops = {}
    for sn in subnames:
        sub = attrs[sn]

        # we don't know how to trim down tables that don't have keys
        if len(sub_keynames[sn]) == 0:
            # automatically keep this entire table
            continue

        # find the used dimensions of the subtable based on its keys
        used_dims = {}
        for dn in sub.dims:
            # limit this dimension based on which values are used by the key coordinates
            # TODO how to find used dimension values of multi-dimension keys?
            for kn in sub_keynames[sn]:
                coord = sub[kn]
                if used_keyvals[kn] is None:
                    continue
                if dn not in coord.dims:
                    continue
                if dn not in used_dims:
                    used_dims[dn] = []
                # find the matching dimension values to the those used key coordinate values
                used = []
                for dimval in sub[dn].values:
                    keyval = coord.sel({dn:[dimval]}).values[0]
                    if keyval in used_keyvals[kn]:
                        used.append(dimval)
                used_dims[dn] += used

            # if this dimension does not appear in any keynames, then assume the entire dimension is used
            if dn not in used_dims:
                used_dims[dn] = sub[dn].values

        # find dimension values that aren't used, to be dropped
        dropvals = {}
        for dn in sub.dims:
            used = used_dims[dn]
            unused = list(filter(lambda v: v not in used, sub[dn].values))
            if len(unused) > 0:
                dropvals[dn] = unused
        # TODO remove
        alldrops[sn] = dropvals

        # drop unused dimension values
        if len(dropvals) > 0:
            attrs[sn] = sub.drop_sel(dropvals)

    # update the global coordinates to reflect the new coordinate values in the subtables
    coords = _build_mxds_coords(mxds, attrs)

    # for kn in keynames:
    #     if used_keyvals[kn] is not None:
    #         print(f"{kn} used values: {np.sort(used_keyvals[kn])}")
    # for sn in alldrops:
    #     print(f"{sn} dropped values: {alldrops[sn]}")

    # create the new mxds
    return xr.Dataset(coords=coords, data_vars=mxds.data_vars, attrs=attrs)