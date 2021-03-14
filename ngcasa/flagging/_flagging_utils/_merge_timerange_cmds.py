# C
import copy


def _merge_timerange_cmds(commands):
    """
    Reduce timerange (antenna + timerange + spw + reason) flagging commands.
    In typical pipeline datasets the time ranges have enough overlaps and/or
    are adjacent such that the number of commands can be reduced by one or two
    orders of magnitude. For example from 1072 down to 38.

    This tries to reproduce the behavior of CASA5/6 flaghelper.parseAgents
    flaghelper_merge_timerangeo. TODO: further cleaning.
    """
    merged = {}
    lunique = []
    for cmd in commands:
        try:
            # skip if no timerange
            if 'time' not in cmd or type(cmd['time']) != slice:
                raise ValueError
            # skip invalid timeranges so they don't remove the whole agent group
            if cmd['time'].stop <= cmd['time'].start:
                raise ValueError

            # sorted list of command keys, excluding time
            compound_key = sorted(x for x in cmd.keys() if x
                                  not in ('time'))
            # compound key of all command keys and their values (e.g. antenna:1)
            compound = tuple((x, cmd[x]) for x in compound_key)

            # merge timerange duplicate compound keys
            try:
                merged[compound]['time'].append(cmd['time'])
            except KeyError:
                merged[compound] = copy.deepcopy(cmd)
        except:
            # add keys merged so far, also on errors like non-hashable keys
            lunique.extend(merged.values())
            # append non mergeable key
            lunique.append(copy.deepcopy(cmd))
            # reset merge to preserve ordering of manual and other flags
            # e.g. summary,manual,manual,manual,summary,manual
            # to summary,merged-manual,summary,manual
            merged = {}

    # add remaining merged keys to non-mergeable keys
    lunique.extend(merged.values())
    if len(lunique) < len(commands):
        print('* Merged {} flagging commands into {}.'.format(len(commands),
                                                              len(lunique)))
    return lunique
