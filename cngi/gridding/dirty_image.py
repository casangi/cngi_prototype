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

def dirty_image(vis_dataset, grid_parms):
    """
    Grids visibilities from Visibility Dataset and returns dirty Image Dataset.
    If to_disk is set to true the data is saved to disk.
    Parameters
    ----------
    vis_xds : xarray.core.dataset.Dataset
        input Visibility Dataset
    grid_parms : dictionary
          keys ('chan_mode','imsize','cell','oversampling','support','to_disk','outfile')
    Returns
    -------
    dirty_image_xds : xarray.core.dataset.Dataset

    """

    import numpy as np
    from numba import jit
    import time
    import math
    import dask.array.fft as dafft
    import xarray as xr
    import dask.array as da
    import matplotlib.pylab as plt
    from .grid import _graph_grid
    import dask.array.fft as dafft
    import dask
    import copy, os
    from numcodecs import Blosc
    from itertools import cycle

    # Parameter adjustments
    grid_parms_dirty_image = copy.deepcopy(grid_parms)
    padding = 1.2  # Padding factor
    dtr = np.pi / (3600 * 180)
    grid_parms_dirty_image['imsize'] = (padding * np.array(grid_parms_dirty_image['imsize'])).astype(int)  # Add padding
    grid_parms_dirty_image['cell'] = np.array(grid_parms_dirty_image['cell']) * dtr
    grid_parms_dirty_image['cell'][0] = -grid_parms_dirty_image['cell'][0]
    grid_parms_dirty_image['to_disk'] = False

    assert grid_parms_dirty_image['chan_mode'] == 'continuum' or grid_parms_dirty_image[
        'chan_mode'] == 'cube', 'The chan_mode parameter in grid_parms can only be \'continuum\' or \'cube\'.'

    grids_and_sum_weights, correcting_cgk_image = _graph_grid(vis_dataset, grid_parms_dirty_image)

    # uncorrected_dirty_image = dafft.fftshift(dafft.ifft2(dafft.ifftshift(grids_and_sum_weights[0], axes=(2,3)), axes=(2,3)), axes=(2,3))
    # uncorrected_dirty_image = uncorrected_dirty_image.real * (grid_parms['imsize'][0] * grid_parms['imsize'][1])
    uncorrected_dirty_image = dafft.fftshift(
        dafft.ifft2(dafft.ifftshift(grids_and_sum_weights[0], axes=(0, 1)), axes=(0, 1)), axes=(0, 1))
    uncorrected_dirty_image = uncorrected_dirty_image.real * (
            grid_parms_dirty_image['imsize'][0] * grid_parms_dirty_image['imsize'][1])

    def correct_image(uncorrected_dirty_image, sum_weights, correcting_cgk):
        sum_weights[sum_weights == 0] = 1
        # corrected_image = (uncorrected_dirty_image/sum_weights[:,:,None,None])/correcting_cgk[None,None,:,:]
        corrected_image = (uncorrected_dirty_image / sum_weights[None, None, :, :]) / correcting_cgk[:, :, None, None]
        return corrected_image

    corrected_dirty_image = da.map_blocks(correct_image, uncorrected_dirty_image, grids_and_sum_weights[1],
                                          correcting_cgk_image)  # ? has to be .data to paralize correctly

    if grid_parms_dirty_image['chan_mode'] == 'continuum':
        freq_coords = [da.mean(vis_dataset.coords['chan'].values)]
        imag_chan_chunk_size = 1
    elif grid_parms_dirty_image['chan_mode'] == 'cube':
        freq_coords = vis_dataset.coords['chan'].values
        imag_chan_chunk_size = vis_dataset.DATA.chunks[2][0]

    chunks = vis_dataset.DATA.chunks
    n_imag_pol = chunks[3][0]
    dirty_image_dict = {}
    coords = {'d0': np.arange(grid_parms_dirty_image['imsize'][0]), 'd1': np.arange(grid_parms_dirty_image['imsize'][1]),
              'chan': freq_coords, 'pol': np.arange(n_imag_pol)}
    dirty_image_dict['CORRECTING_CGK'] = xr.DataArray(da.array(correcting_cgk_image), dims=['d0', 'd1'])
    # dirty_image_dict['VIS_GRID'] = xr.DataArray(grids_and_sum_weights[0], dims=['chan','pol','u', 'v'])
    dirty_image_dict['VIS_GRID'] = xr.DataArray(grids_and_sum_weights[0], dims=['d0', 'd1', 'chan', 'pol'])
    dirty_image_dict['SUM_WEIGHT'] = xr.DataArray(grids_and_sum_weights[1], dims=['chan', 'pol'])
    dirty_image_dict['DIRTY_IMAGE'] = xr.DataArray(corrected_dirty_image, dims=['d0', 'd1', 'chan', 'pol'])
    dirty_image_xds = xr.Dataset(dirty_image_dict, coords=coords)

    if grid_parms['to_disk'] == True:
        outfile = grid_parms['outfile']
        tmp = os.system("rm -fr " + outfile)
        tmp = os.system("mkdir " + outfile)

        compressor = Blosc(cname='zstd', clevel=2, shuffle=0)
        encoding = dict(zip(list(dirty_image_xds.data_vars), cycle([{'compressor': compressor}])))
        start = time.time()
        xr.Dataset.to_zarr(dirty_image_xds, store=outfile, mode='w', encoding=encoding)
        dirty_image_time = time.time() - start
        print('Dirty Image time ', time.time() - start)

        dirty_image_xds = xr.open_zarr(outfile)
        dirty_image_xds.attrs['dirty_image_time'] = dirty_image_time
        return dirty_image_xds
    else:
        return dirty_image_xds
