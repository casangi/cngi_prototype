#   Copyright 2020-21 European Southern Observatory, ALMA partnership
#   Copyright 2020 AUI, Inc. Washington DC, USA
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
