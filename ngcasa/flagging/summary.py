# C
import dask
import xarray as xr
from ._flagging_utils._summary_utils import _pol_id_to_corr_type_name, _cast_int


def summary(mxds, xds_idx, flag_varname='FLAG'):
    """
    Produces a summary with counts of flags from a flag variable of an xarray dataset.
    The summary is returned as a dictionary with the same structure as the dictionary
    returned by CASA6/flagdata in summary mode. It includes total counts and partial
    counts broken down by array, observation, correlation, field, scan, and antenna.

    Note: per-SPW counts are not included in this prototype version. Handling of
    multi-SPWs operations is subject to changes depending on interface design at (or
    around) the ngCASA layer.

    Parameters
    ----------
    mxds: xarray.core.dataset.Dataset
        Dataset of xarray datasets
    xds_idx: int
        Index of the xarray datasets to get counts from (index in the xds'i'
        attributes of mxds)
    flag_varname: str
        Name of the flag variable to summarize
    TBD - tolerance

    Returns:
    -------
    dict
        Dictionary with counts of flags set and total flags, for the whole dataset
        also broken down (grouped by) by multiple criteria (scan, field, antenna,
        correlation, etc.)
    """
    # mxds is required to grab meta-information (names of fields, antennas and
    # correlations). For now, working only on 1 SPW (given by index)

    xds = mxds.attrs['xds' + '{}'.format(xds_idx)]
    return _summary_groupby(mxds, xds, flag_varname)


def _summary_groupby(mxds, xds, flag_varname):
    # an implementation of summary based mostly on xarray groupby
    result = {}

    # Assumes: xds['presence_baseline'] = xds.DATA.notnull().sum(['chan'])
    # To use presence_baseline.sum() as count of 'total'
    flag_var = xds[flag_varname].sum(['chan']).where(xds.presence_baseline > 0, other=0)

    # Sum up chan dimension, which is not relevant for any of the following:
    # FIELD, OBS, ANT, etc.
    pres_var = xds['presence_baseline']  # assume it's been .sum(['chan'])

    # For joint grouping of FLAG data var and presence_baseline
    # From FLAG vars we'll get the 'flagged' counters. From presence_baseline we get
    # the 'total' counters.
    count_xds = xr.Dataset({'flag_var': flag_var, 'presence_baseline': pres_var})

    _count_pol_and_totals(result, xds, count_xds)

    _count_array_scan_obs(result, xds, count_xds)

    _count_field(result, xds, count_xds, mxds.fields.values)

    _count_antenna(result, xds, count_xds, mxds.ANTENNA.NAME.values)

    res_comp = dask.compute(result)
    return _cast_int(res_comp[0])


def _count_pol_and_totals(result, xds, count_xds):
    # Count by pol/correlation + grand totals
    grand_total = 0
    grand_flagged = 0
    result['correlation'] = {}
    groups = count_xds.groupby(xds.pol)

    # groups = count_xds.flag_var.groupby(xds.pol)
    for grp_idx, grp_ds in groups:
        # total = nchan * grp_ds.count()  # would include the non-present time-baselines
        flagged = grp_ds.flag_var.sum()
        # flagged = grp_ds.sum()
        total = grp_ds.presence_baseline.sum()
        # total = xds.chan.count() * grp_ds.shape[0] * grp_ds.shape[1]
        corr_str = _pol_id_to_corr_type_name(grp_idx)
        result['correlation']['{}'.format(corr_str)] = {'flagged': flagged, 'total': total}
        grand_total += total
        grand_flagged += flagged

    result['total'] = grand_total
    result['flagged'] = grand_flagged


def _count_array_scan_obs(result, xds, count_xds):
    # Straightforward grouping for counts by data variables (ARRAY_ID, SCAN_NUMBER,
    # OBSERVATION_ID)
    grouping_names_vars = [('array', xds.ARRAY_ID),
                           ('scan', xds.SCAN_NUMBER),
                           ('observation', xds.OBSERVATION_ID)]
    for name, data_var in grouping_names_vars:
        result[name] = {}
        groups = count_xds.groupby(data_var)
        for grp_lbl, grp_ds in groups:
            if grp_lbl < 0:
                continue
            flagged = grp_ds.flag_var.sum()
            # ? total = nchan * grp_ds.count() <- total non-nan 'after last appplyflags'
            total = grp_ds.presence_baseline.sum()
            group_str = '{}'.format(grp_lbl)
            result[name][group_str] = {'flagged': flagged, 'total': total}


def _count_field(result, xds, count_xds, field_names):
    # Counting by field name, but grouping by FIELD_ID. Multiple field IDs can
    # map to a same name (object, target)
    result['field'] = {}
    groups = count_xds.groupby(xds.FIELD_ID)
    for grp_idx, grp_ds in groups:
        if grp_idx < 0:
            continue
        flagged = grp_ds.flag_var.sum()
        # ? total = nchan * grp_ds.count() <- total non-nan 'after last applyflags'
        total = grp_ds.presence_baseline.sum()
        field_str = '{}'.format(field_names[grp_idx])
        if field_str in result['field']:
            result['field'][field_str]['flagged'] += flagged
            result['field'][field_str]['total'] += total
        else:
            result['field'][field_str] = {'flagged': flagged, 'total': total}


def _count_antenna(result, xds, count_xds, ant_names):
    # Counting by antenna (can be ANTENNA1 or ANTENNA2)
    result['antenna'] = {}
    groups_ant1 = count_xds.groupby(xds.ANTENNA1)
    groups_ant2 = count_xds.groupby(xds.ANTENNA2)
    for groups in [groups_ant1, groups_ant2]:
        for grp_idx, grp_ds in groups:
            if grp_idx < 0:
                continue
            flagged = grp_ds.flag_var.sum()
            # ? total = nchan * grp_ds.flag_var.count()
            #       <- total non-nan 'after last applyflags'
            total = grp_ds.presence_baseline.sum()
            ant_str = '{}'.format(ant_names[grp_idx])
            if ant_str in result['antenna']:
                result['antenna'][ant_str]['flagged'] += flagged
                result['antenna'][ant_str]['total'] += total
            else:
                result['antenna'][ant_str] = {'flagged': flagged, 'total': total}
