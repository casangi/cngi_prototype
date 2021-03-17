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

########################
def join_vis(mxds, vis1, vis2):
    """
    Concatenate together two Visibility xds's of compatible shape from the same mxds


    Extended Summary
    ----------------
    The data variables of the two datasets are merged together, with some
    limitations (see "Current Limitations" in the Notes section).

    Coordinate values that are not also being used as dimensions are compared for
    equality.

    Certain known attributes are updated, namely "ddi". For the rest, they are
    merged where keys are present in one dataset but not the other, or the values
    from the first dataset override those from the second where the keys are the
    same.


    Parameters
    ----------
    mxds : xarray.core.dataset.Dataset
        input multi-xarray Dataset with global data
    vis1 : str
        first visibility partition in the mxds to join
    vis2: str
        second visibility partition in the mxds to join


    Returns
    -------
    xarray.core.dataset.Dataset
        New output multi-xarray Dataset with global data


    Warnings
    --------
    Joins are highly discouraged for datasets that don't share a common 'global' DDI
    (ie are sourced from different .zarr archives). Think really hard about if a
    join would even mean anything before doing so.

    Warnings
    --------
    DDIs are separated by spectral window and correlation (polarity) because it is a
    good indicator of how data is gathered in hardware. Ultimately, if source data
    comes from different DDIs, it means that the data followed different paths
    through hardware during the measurement. This is important in that there are
    more likely to be discontinuities across DDIs than within them. Therefore,
    don't haphazardly join DDIs, because it could break the inherent link between
    data and hardware.


    Notes
    -----
    **Conflicts in data variable values between datasets:**

    There are many ways that data values could end up differing between datasets for
    the same coordinate values. One example is the error bars represented in SIGMA
    or WEIGHT could differ for a different spw.

    There are many possible solutions to dealing with conflicting data values:

    1. only allow joins that don't have conflicts (current solution)
    2. add extra indexes CHAN, POL, and/or SPW to the data variables that conflict
    3. add extra indexes CHAN, POL, and/or SPW to all data variables
    4. numerically merge the values (average, max, min, etc)
    5. override the values in xds1 with the values in xds2

    **Joins are allowed for:**

    Datasets that have all different dimension values.

    - Example: xds1 covers time range 22:00-22:59, and xds2 covers time range 23:00-24:00

    Datasets that have overlapping dimension values with matching data values at all of those coordinates.

    - Example: xds1.PROCESSOR_ID[0][0] == xds2.PROCESSOR_ID[0][0]


    **Current Limitations:**

    Joins are not allowed for datasets that have overlapping dimension values with mismatched data values at any of those coordinates.

    - Example: xds1.PROCESSOR_ID[0][0] != xds2.PROCESSOR_ID[0][0]
    - See "Conflicts in data variable values", above

    Joins between 'global' datasets, such as those returned by cngi.dio.read_vis(ddi='global'), are probably meaningless and should be avoided.
    Datasets do not need to have the same shape.

    - Example xds1.DATA.shape != xds2.DATA.shape


    Examples
    --------
    ### Use cases (some of them), to be turned into examples. Note: these use cases come from CASA's mstransform(combinespws=True) and may not apply to ddijoin.

    - universal calibration across spws
    - autoflagging with broadband rfi
    - uvcontfit and uvcontsub
    - joining datasets that had previously been split, operated on, and are now being re-joined
    """

    import xarray.core.merge
    import cngi._utils._specials as specials
    from cngi._utils._io import mxds_copier

    # get some values
    func_name = join_vis.__name__
    xds1 = mxds.attrs[vis1]
    xds2 = mxds.attrs[vis2]

    # verify that the non-dimension coordinates are the same in both datasets
    # Example for why we do this check: joining two vis datasets with mismatched "antenna" coordinates is probably invalid.
    for coord_name in xds1.coords:
        # skip dimension coordinates, errors in these coordinates will be caught by xr.merge
        if coord_name in xds1.dims or coord_name in xds2.dims:
            continue
        # skip missing coordinates
        if not coord_name in xds2.coords:
            continue
        # verify that the coordinates match
        assert((xds1.coords[coord_name] == xds2.coords[coord_name]).all()), f"######### ERROR: coordinate \"{coord_name}\" mismatch between datasets"

    # perform the data_vars and dimension coordinates merge
    try:
        xds_merged = xr.merge([xds1, xds2], compat='no_conflicts', combine_attrs='override')
    except xarray.core.merge.MergeError as e:
        raise RuntimeError(f"{func_name} can't join datasets with conflicting data variable values at overlapping indexes. See help({func_name}) for more info.") from e

    # copy any attributes from xds2 that don't exist in xds1
    # xds_merged.attrs = xds_merged.attrs.copy() # xr.merge does not apparently make a copy of the attributes
    for attr_name in xds2.attrs:
        if attr_name in xds1.attrs:
            continue
        xds_merged.attrs[attr_name] = xds2.attrs[attr_name]

    # set the special attributes
    for attr_name in specials.attrs():
        if not attr_name in xds_merged.attrs:
            continue
        elif attr_name == "ddi":
            xds_merged.attrs[attr_name] = f"{func_name}({xds1.attrs['ddi']},{xds2.attrs['ddi']})"
        else:
            raise RuntimeError(f"Programmer error: did not anticipate the special attribute {attr_name}! Don't know how to join this attribute!")

    new_mxds = mxds_copier(mxds, vis1, xds_merged)
    new_mxds.attrs.pop(vis2)
    return new_mxds
