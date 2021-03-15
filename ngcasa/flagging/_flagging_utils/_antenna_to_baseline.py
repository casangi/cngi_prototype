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


def _antenna_to_baseline(mxds, xds, ant_name):
    def ant_name_to_idx(mxds, ant_name):
        # just want to find the first index. Could be a numba jitted for np.ndenumerate
        idxs = np.where(mxds.antennas == ant_name)
        if not idxs or 0 == len(idxs) or 0==len(idxs[0]):
            raise RuntimeError(f"Could not find antenna: {ant_name} in dataset")
        return idxs[0][0]

    def ant_idx_to_baseline_idxs(xds, ant_idx):
        baselines_ant1 = np.where(xds.ANTENNA1.values == ant_idx)
        baselines_ant2 = np.where(xds.ANTENNA2.values == ant_idx)
        baseline_idxs = np.concatenate((baselines_ant1[0], baselines_ant2[0]))
        return baseline_idxs

    ant_idx = ant_name_to_idx(mxds, ant_name)
    baseline_idxs = ant_idx_to_baseline_idxs(xds, ant_idx)

    return baseline_idxs
