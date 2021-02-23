# C
"""
this module will be included in the api
"""
import xarray as xr
from ._flagging_utils._parse_sel_cmd import _parse_sel_cmd
from ._flagging_utils._read_flagcmds import _read_flagcmds

def manual_flag(mxds, xds_idx, commands=None, cmd_filename=None):
    """
    Implements the 'manual' flagging method (equivalent to CASA6's flagdata manual
    mode).

    Parameters
    ----------
    mxds: xarray.core.dataset.Dataset
        Input Dataset
    xds_idx: int
        Index of the xarray datasets to get counts from (index in the xds'i'
        attributes of mxds). This is an oversimplification (early prototyping)
    commands: List[Dict]
        List of selection commands. Each item in the list represent one
        selection of data to flag.
        Every selection item is a dictionary. Selection is currently supported
        by 'time', 'chan', 'antenna', and 'pol'. 'time', 'chan', and 'pol'
        correspond directly to the dimensions of FLAG and DATA varaibles in
        the xarray datasets. 'antenna' is translated to the 'baseline' dimension.
    cmd_filename: filename
        Name of a file with text flagging commands, using the same format as
        used in the CASA6 pipelines for the "*flagonline.txt" or
        "*flagcmds.txt" files.
    TBD - Additional selection parameters / criteria

    Returns:
    -------
    xarray.core.dataset.Dataset
        Dataset with flags set on the selections given in commands and/or
        cmd_filename
    """
    if commands and not isinstance(commands, list):
        raise ValueError('Parameter selection must be a list of selection dicts')
    all_cmds = []
    if commands:
        all_cmds.extend(commands)
    if cmd_filename:
        all_cmds.extend(_read_flagcmds(cmd_filename))
    if not all_cmds:
        raise RuntimeError("No valid flagging commands found in inputs. Direct "
                           f"command list is: {commands}. File is: {cmd_filename}")
    xds = mxds.attrs['xds' + '{}'.format(xds_idx)]
    return _manual_with_reindex_like(mxds, xds, all_cmds)


def _manual_with_reindex_like(mxds, xds, cmds):
    flag_var = 'FLAG'
    ret_xds = xds.assign()
    for cmd in cmds:
        selection = _parse_sel_cmd(mxds, xds, cmd)
        fsel = xds[flag_var].sel(selection)
        if 0 in fsel.shape:
            print('WARNING: selection results in 0 shape. Sel: {}. Shape: {}'.
                  format(fsel, fsel.shape))
            continue

        # Alternative 1 with .sel(): sel-ones, broadcast, or
        flag_slice = xr.ones_like(fsel, dtype=bool)
        reindexed_slice = flag_slice.reindex_like(xds[flag_var], fill_value=False)
        ret_xds[flag_var] |= reindexed_slice

    return ret_xds
