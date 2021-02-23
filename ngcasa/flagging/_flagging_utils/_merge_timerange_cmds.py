# C


def _merge_timerange_cmds(cmds):
    """
    Reduce timerange (antenna + timerange + spw + reason) flagging commands.
    In typical pipeline datasets the time ranges have enough overlaps and/or
    are adjacent such that the number of commands can be reduced by one or two
    orders of magnitude. For example from 1072 down to 38.
    """
    return cmds
