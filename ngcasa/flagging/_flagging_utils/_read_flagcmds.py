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
import numpy as np
from ._merge_timerange_cmds import _merge_timerange_cmds


def _read_flagcmds(filename, spw_name=None):
    """
    Limited version of read_flagcmds which can only read online flags as usually
    included in "*.flagonline.txt", "*.flagcmds.txt" files produced by
    importasdm and the pipelines (example: uid___A002_Xc25b24_X599.flagonline.txt).

    Usage:
    cmds = read_flagcmds.read_flagcmds('uid___A002_Xc25b24_X599.flagonline.txt', 
                                       vis.name, ant_name_to_idx)
    ant_names:
    * before: ant_name_to_idx = 
       {name: idx for idx, name in enumerate(vis_global.ANT_NAME.values)
    * now: ant_name_to_idx = 
    {name: idx for idx, name in enumerate(vis_global.antennas.values)}
       (or vis_global.ANT.NAME.values)

    For the moment it only recognizes 'antenna=', 'timerange=' and 'spw=' selections.
    If no spw_name is given, it will blatantly ignore the spw= selection, assuming all
    commands apply to the current SPW/xds being flagged.    
    """
    def parse_line(line, spw_name):
        """ 
        Turn a line from a "*.flagsonline.txt" file into a selection command for
        manual_flag.
        Returns none if the selection excludes this spw_name
        """
        params = line.split(' ')
        out = {}
        for par in params:
            key, value = par.split('=')
            if key == 'antenna':
                value = value.strip("'")
                ant_split = value.split('&')
                ant_name = ant_split[0]
                out['antenna'] = ant_name
            elif key == 'timerange':
                # Simplified conversion CASA_format_str->datetime64
                # CASA tstamp format: '2017/07/20/03:21:53.328'
                # ngCASA tstamp format: '2017-07-20T03:24:36.671999931'
                value = value.strip("'")
                value = value.replace('/', '-')
                start, end = value.split('~')
                start = start[0:10] + 'T' + start[11:] + '0'
                end = end[0:10] + 'T' + end[11:] + '0'
                out['time'] = slice(np.datetime64(start), np.datetime64(end))
            elif key == 'spw':
                if spw_name and spw_name not in value:
                    print(f"WARNING: ignoring, not selected spw: {spw_name}")
                    return None
        return out

    cmds = []
    with open(filename) as cmdf:
        lines = cmdf.readlines()
        for line in lines:
            strp = line.strip()
            if not strp or strp.startswith('#'):
                continue
            item = parse_line(strp, spw_name)
            if item:
                cmds.append(item)

    return _merge_timerange_cmds(cmds)
