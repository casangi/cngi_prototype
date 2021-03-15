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
import xarray as xr
import typing

###############################################
def split_dataset(mxds : xr.Dataset, xds_names : typing.Union[str, typing.List[str]]) -> xr.Dataset:
    """Pull the xds visibilites out with the mxds, preserving only that information in the subtables that is related to the given visibilities.

    Extended Summary
    ----------------
    Creates a new mxds to return based off the input mxds. Only the visibilites
    mentioned in xds_names are included. Subtable data is reduced to only
    include related information, based on the relational keys in the visibility
    tables. Finally, the coordinate values of the new mxds are updated to
    reflect the limited coordinate values in the included visibilities.

    Parameters
    ----------
    mxds: xarray.Dataset
        The multi-xds dataset to pull data out of.
    xds_names: str or list
        Name(s) of the visibilities dataset. Each name should be of the form "xds*"

    Returns
    -------
    xarray.Dataset
        A new mxds, which includes just the xds_names visibility Dataset(s) and
        the related information from the mxds subtables.
    """
    import numpy as np
    from cngi._utils._join_split import get_subtable_dimcoords_or_primcoords, gen_keyname_variants, build_mxds_coords
    from cngi._utils._mxds_ops import get_subtables

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
        sub_kns = get_subtable_dimcoords_or_primcoords(attrs[sn], sn)
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
    coords = build_mxds_coords(mxds, attrs)

    # for kn in keynames:
    #     if used_keyvals[kn] is not None:
    #         print(f"{kn} used values: {np.sort(used_keyvals[kn])}")
    # for sn in alldrops:
    #     print(f"{sn} dropped values: {alldrops[sn]}")

    # create the new mxds
    return xr.Dataset(coords=coords, data_vars=mxds.data_vars, attrs=attrs)