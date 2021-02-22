# C
import numpy as np


def _antenna_to_baseline(mxds, xds, ant_name):
    def ant_name_to_idx(mxds, ant_name):
        # just want to find the first index. Could be a numba jitted for np.ndenumerate
        idxs = np.where(mxds.antennas == ant_name)
        return idxs[0][0]

    def ant_idx_to_baseline_idxs(xds, ant_idx):
        baselines_ant1 = np.where(xds.ANTENNA1.isel(time=0).values == ant_idx)
        baselines_ant2 = np.where(xds.ANTENNA2.isel(time=0).values == ant_idx)
        baseline_idxs = np.concatenate((baselines_ant1[0], baselines_ant2[0]))
        return baseline_idxs

    ant_idx = ant_name_to_idx(mxds, ant_name)
    baseline_idxs = ant_idx_to_baseline_idxs(xds, ant_idx)
    return baseline_idxs
