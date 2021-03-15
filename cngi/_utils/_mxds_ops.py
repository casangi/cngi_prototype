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

import xarray as xr
import typing

def get_subtable_primary_key_names(xds: xr.Dataset, subtable_name : str) -> typing.List[str]:
    """Finds the coordinates that are probably primary keys in the given subtable"""
    ret = [] # type: typing.List[str]
    for coord_name in list(xds.coords):
        if subtable_name.lower() in coord_name.lower():
            if "_id" in coord_name.lower():
                ret.append(coord_name)
    return ret

def get_subtables(mxds : xr.Dataset) -> typing.List[str]:
    """Finds the attributes that are probably subtables in the given mxds."""
    subtable_names = []
    for subtable_name in list(mxds.attrs):
        # don't include visibility tables
        if "xds" in subtable_name:
            continue
        # don't include anything that isn't a subtable
        if not isinstance(mxds.attrs[subtable_name], xr.Dataset):
            continue
        # include this attribute as a subtable
        subtable_names.append(subtable_name)

    return subtable_names

def assign_dimensions_for_primary_coordinates(sub0 : xr.Dataset, sub1 : xr.Dataset, subtable_name : str) -> typing.Tuple[xr.Dataset, xr.Dataset]:
    """Assigns the primary key coordinates as dimensions for those that don't have dimension coordinates."""
    import re
    import numpy as np

    # get the shared primary key coordinates that aren't dimension coordinates
    prime0 = get_subtable_primary_key_names(sub0, subtable_name)
    prime1 = get_subtable_primary_key_names(sub1, subtable_name)
    shared_primes = [x for x in prime0 if x in prime1] # intersection of prime0 and prime1
    shared_primes = list(filter(lambda x: x not in sub0.dims and x not in sub1.dims, shared_primes)) # exclude current dimension coordinates

    # verify that the primary key coordinates have a single dimension
    good_shared_primes = []
    anon = re.compile("d[0-9]")
    for coord in shared_primes:
        if (len(sub0[coord].dims) != 1) or (len(sub1[coord].dims) != 1):
            print(f"Warning: can't assign primary key coordinate {coord} as dimension coordinate in {assign_dimensions_for_primary_coordinates.__name__}: too many dimensions.")
            continue
        first_dim = sub0[coord].dims[0]
        if not anon.match(first_dim):
            print(f"Warning: can't assign primary key coordinate {coord} as dimension coordinate in {assign_dimensions_for_primary_coordinates.__name__}: first dimension {first_dim} is not anonymous.")
            continue
        if len(np.unique(sub0[coord].values)) != len(sub0[coord].values):
            if (subtable_name != "FEED") or (coord != "feed_id"): # feed_id in most cases should NOT be unique
                print(f"Warning: can't assign primary key coordinate {coord} as dimension coordinate in {assign_dimensions_for_primary_coordinates.__name__}: coordinate does not have all-unique values.")
            continue
        good_shared_primes.append(coord)
    shared_primes = good_shared_primes

    # assign dimension coordinates
    ret = [sub0, sub1]
    for subi in range(2):
        sub = sub0 if subi == 0 else sub1
        for coord in shared_primes:
            first_dim = sub[coord].dims[0]
            if (coord not in sub.dims) and (first_dim not in sub.coords):
                sub = sub.rename({first_dim: coord}) # replace the non-dimensional coordinate with the primary key name
                # Renaming a dimension to the same name as a coordinate does not do whatever happens internally to mark that coordinate as a dimensional coordinate.
                # You wouldn't know this based off of print(sub), because the print statement adds a "*" next to the coordinate name.
                # Symptom: sub0.broadcast_like(sub1) with different values in their "coord" coordinate will not broadcast.
                # Example: change the values of "source_id" in the "SOURCE" subtable after renaming, here, then try to broadcast against each other.
                # Workaround: create a new dataset with all the same values. This corrects whatever issues there are internally.
                # TODO find encapsulated use case and submit bug report to xarray
                sub = xr.Dataset(coords=sub.coords, data_vars=sub.data_vars, attrs=sub.attrs)
        ret[subi] = sub

    return (ret[0], ret[1])

def check_mxds_subtable_ref_ids(mxds : xr.Dataset, onerror="print_warning"):
    """!!!COMPUTES!!! verifies that the references to primary keys between subtables exist.

    Parameters
    ----------
    mxds: xarray.Dataset
        The multi-visibility dataset containing subtables to check.
    onerror: str
        What to do when bad values are found. Options are "print_warning", "raise", or
        any other value to do nothing but return the subtable in the returned list.

    Returns
    -------
    list[str]
        The list of subtables with bad reference values. Empty if no bad values are found.
    """
    import numpy as np
    ret = []

    subtable_names = get_subtables(mxds)
    for subtable_name0 in subtable_names:
        sub_pkname0 = subtable_name0.lower().replace("spectral_window", "spw")
        sub0 = mxds.attrs[subtable_name0]

        # get the subtable primary "key" id(s)
        primary_key_names = get_subtable_primary_key_names(sub0, subtable_name0)

        # check against references in other subtables
        for subtable_name1 in subtable_names:
            # don't compare to itself
            if subtable_name0 == subtable_name1:
                continue
            sub1 = mxds.attrs[subtable_name1]

            # check coordinates and data_vars for references
            cdnames = list(sub1.coords) + list(sub1.data_vars) # type: typing.List[str]
            for cdname in cdnames:
                # only match against primary "key" references
                if sub_pkname0 not in cdname.lower():
                    continue
                if "_id" not in cdname.lower():
                    continue

                # compare the values in sub1[cdname] to the similar primary key from sub0
                for pkname in primary_key_names:
                    if (pkname.lower() not in cdname.lower()) and \
                       (cdname.lower() not in pkname.lower()):
                        continue

                    # check all values for existance
                    if cdname in sub1.coords:
                        vals = sub1.coords[cdname].values.flatten()
                    else:
                        vals = sub1.data_vars[cdname].values.flatten()
                    vals = np.unique(vals)
                    for v in vals:
                        if v in sub0.coords[pkname]:
                            # value is in the primary key: ok
                            continue
                        # value is not in the primary key: not ok
                        ret.append(subtable_name1)
                        msg = f"Warning: reference value {v} in subtable {subtable_name1} does not exist in {subtable_name0}.{pkname}!"
                        if onerror == "print_warning":
                            print(msg)
                        elif onerror == "raise":
                            raise ValueError(msg)

    # done checking reference values, return the results
    ret = list(np.unique(ret))
    return ret