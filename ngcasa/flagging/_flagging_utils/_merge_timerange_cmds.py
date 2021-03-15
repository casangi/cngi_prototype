#  CASA Next Generation Infrastructure
#  Copyright (C) 2021 AUI, Inc. Washington DC, USA
#  Copyright (C) 2021 European Southern Observatory, ALMA partnership
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
