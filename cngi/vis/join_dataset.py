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

###############################################
def join_dataset(mxds1, mxds2):
    """
    Join together two visibility zarr directories.


    Extended Summary
    ----------------
    Creates a new mxds with all the visibilities "xds*" of mxds1/mxds2 and all
    the subtables of mxds1 and mxds2. Visibilities are renamed so as not to
    collide where necessary. Subtable values are preserved where they are
    equal, and updated to have new dimensional coordinate values where they are
    not equal.

    The order of the visibilities in the two mxds is preserved. If mxds1 and
    mxds2 have visibilits ["xds0", "xds2"] and ["xds1", "xds2"], respectively,
    then the new mxds will have visibilities ["xds0", "xds2", "xds3", "xds4"],
    where the visibilities from mxds2 got renamed xds1->xds3 and xds2->xds4.

    Parameters
    ----------
    mxds1 : xarray.core.dataset.Dataset
        First multi-xarray Dataset with global data to join.
    mxds2 : xarray.core.dataset.Dataset
        Second multi-xarray Dataset with global data to join.

    Returns
    -------
    xarray.core.dataset.Dataset
        New output multi-xarray Dataset with global data.
    """
    from cngi._utils._join_split import append_mxds_subtables, gen_keyname_variants, apply_coord_remap, apply_data_var_remap
    from cngi._utils._io import mxds_copier
    import re

    # create the new mxds
    mxds, keyname_maps = append_mxds_subtables(mxds1, mxds2)

    # Find the index of the last visibility, used to rename visibilities from
    # mxds2 to mxds1.
    visnames1 = list(filter(lambda x: "xds" in x, mxds1.attrs))
    visnames2 = list(filter(lambda x: "xds" in x, mxds2.attrs))
    numreg = re.compile("[0-9]+")
    lastindex = 0
    for visname in reversed(sorted(visnames1 + visnames2)): # type: str
        if len(numreg.findall(visname)) > 0:
            lastindex = int(numreg.findall(visname)[0])
            break
    nextindex = lastindex+1

    # add in the visibilities from mxds2
    for aname in sorted(visnames2): # type: str
        main = mxds2.attrs[aname]

        # update keyname reference values
        for kn in keyname_maps:
            map_func = lambda x: x if x not in keyname_maps[kn] else keyname_maps[kn]
            for knvariant in gen_keyname_variants(kn):
                if knvariant in main.coords:
                    main = apply_coord_remap(main, knvariant, map_func)
                if knvariant in main.data_vars:
                    main = apply_data_var_remap(main, knvariant, map_func)

        # Preserve the visibility name where possible.
        if len(numreg.findall(aname)) > 0:
            visidx = int(numreg.findall(aname)[0])
            if (f"xds{nextindex}" not in mxds.attrs) and (visidx >= nextindex):
                nextindex = visidx

        # Insert into the returned xds.
        while f"xds{nextindex}" in mxds.attrs:
            nextindex += 1
        mxds.attrs[f"xds{nextindex}"] = main
        nextindex += 1

    return mxds

