#   Copyright 2019 AUI, Inc. Washington DC, USA
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
"""
this module will be included in the api
"""

########################
def ddijoin(xds1, xds2):
    """
    Concatenate together two Visibility Datasets of compatible shape

    Extended Summary
    ----------------
    The data variables of the two datasets are merged together, with some
    limitations (see the Notes section).

    Coordinates are compared for equality, and certain known attributes are updated,
    namely "ddi". For the rest, the values from the second dataset override those
    from the first.

    Parameters
    ----------
    xds1 : xarray.core.dataset.Dataset
        first Visibility Dataset to join
    xds2 : xarray.core.dataset.Dataset
        second Visibility Dataset to join

    Returns
    -------
    xarray.core.dataset.Dataset
        New Visibility Dataset with combined contents
    
    Warnings
    --------
    Joins are highly discouraged for datasets that don't share a common 'global' DDI
    (ie are sourced from different .zarr archives). Think really hard about if
    that even means anything before doing so.

    DDIs are separated by spectral window and correlation (polarity) because it is a
    good indicator of how data is gathered in hardware. Ultimately, if source data
    comes from different DDIs, it means that the data followed different paths
    through hardware during the measurement. This is important in that there are
    more likely to be discontinuities across DDIs than within them. Therefore,
    don't haphazardly join DDIs, because it could break the inherent link between
    data and hardware.

    Notes
    -----
    *Conflicts in data variable values between datasets:*
    There are many ways that data values could end up differing between datasets for
    the same coordinate values. One example is the error bars represented in SIGMA
    or WEIGHT could differ for a different spw.
    There are many possible solutions to dealing with conflicting data values:
    A) only allow joins that don't have conflicts (current solution)
    B) add extra indexes CHAN, POL, and/or SPW to the data variables that conflict
    C) add extra indexes CHAN, POL, and/or SPW to all data variables
    D) numerically merge the values (average, max, min, etc)
    E) override the values in xds1 with the values in xds2

    *Current limitations:*
    Joins are allowed for datasets that have all different coordinate values.
        Example: xds1 covers time range 22:00-22:59, and xds2 covers time range 23:00-24:00
    Joins are allowed for datasets that have overlapping coordinate values with matching data values at all of those coordinates.
        Example: xds1.PROCESSOR_ID[0][0] == xds2.PROCESSOR_ID[0][0]
    Joins are not allowed for datasets that have overlapping coordinate values with mismatched data values at any of those coordinates.
        Example: xds1.PROCESSOR_ID[0][0] != xds2.PROCESSOR_ID[0][0]
        See "Conflicts in data variable values", above
    Joins between 'global' datasets, such as those returned by cngi.dio.read_vis(ddi='global'), are probably meaningless and should be avoided.
    Datasets must have the same shape.
        Example xds1.DATA.shape == xds2.DATA.shape

    Examples
    --------
    ### Use cases (some of them), to be turned into examples:
    # universal calibration across spws
    # autoflagging with broadband rfi
    # uvcontfit and uvcontsub
    # joining datasets that had previously been split, operated on, and are now being re-joined
    """
    return {}
