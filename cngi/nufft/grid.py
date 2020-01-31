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

def grid(vis_dataset,grid_parms):
      """
      
      Parameters
      ----------
      vis_dataset : xarray dataset
      
      grid_parms : dictionary 
          ('imsize','cell','oversampling','support')

      Returns
      -------
      grid_dataset : xarray dataset
      """
      
      import xarray as xr
      import numpy as np
      import time
      import math
      import dask.array as da
      from cngi.nufft import gridding_convolutional_kernels as gck
      from  cngi.nufft import standard_gridder as sg
      import sparse
      
      padding = 1.2  # Padding factor
      grid_parms['imsize'] = (padding * grid_parms['imsize']).astype(int) # Add padding
      
      # Creating gridding kernel
      # The support in CASA is defined as the half support, which is 3
      cgk, correcting_cgk_image = gck.create_prolate_spheroidal_kernel(grid_parms['oversampling'], grid_parms['support'], grid_parms['imsize'])
      cgk_1D = gck.create_prolate_spheroidal_kernel_1D(grid_parms['oversampling'], grid_parms['support'])
      
      #Getting data for gridding
      n_chan_in_chunk = vis_dataset.DATA.chunks[2][0]
      
      freq_chan = da.from_array(vis_dataset.coords['chan'].values,chunks=(n_chan_in_chunk))
      
      n_pol_in_chunk = vis_dataset.DATA.chunks[3][0]
      if grid_parms['chan_mode'] == 'cube':
          n_cube_chan = vis_dataset.DATA.chunks[2][0]
          output_dims = ("n_time", "n_baseline", "n_chan", "n_pol", "n_cube_chan", "n_pol_in_chunk", "n_u", "n_v", "n_switch")
          new_axes = {"n_switch": 2, "n_u": grid_parms['imsize'][0], "n_v": grid_parms['imsize'][1], "n_cube_chan" : n_cube_chan, "n_pol_in_chunk" : n_pol_in_chunk}
      else: #continuum
          output_dims = ("n_time", "n_baseline", "n_chan", "n_pol", "n_continuum_chan", "n_pol_in_chunk", "n_u", "n_v", "n_switch")
          new_axes = {"n_switch": 2,"n_u": grid_parms['imsize'][0], "n_v": grid_parms['imsize'][1], "n_continuum_chan" : 1, "n_pol_in_chunk" : n_pol_in_chunk}
      
      start = time.time()
      grids_and_sum_weights = da.blockwise(sg.standard_grid_dask_sparse, output_dims,
                                         vis_dataset.DATA, ("n_time", "n_baseline", "n_chan", "n_pol"),
                                         vis_dataset.UVW, ("n_time", "n_baseline", "uvw"), 
                                         vis_dataset.WEIGHT, ("n_time", "n_baseline", "n_pol"), 
                                         vis_dataset.FLAG_ROW,("n_time", "n_baseline"), 
                                         vis_dataset.FLAG, ("n_time", "n_baseline", "n_chan", "n_pol"),
                                         freq_chan, ("n_chan",),
                                         new_axes=new_axes,
                                         adjust_chunks={"n_time": 1, "n_baseline": 1, "n_chan": 1, "n_pol": 1},
                                         cgk_1D=cgk_1D, grid_parms=grid_parms,
                                         dtype=complex)
      
      
      if grid_parms['chan_mode'] == 'continuum':
          grids_and_sum_weights = (grids_and_sum_weights.sum(axis=(0,1,2,3)))
          grids_and_sum_weights = grids_and_sum_weights.compute()
          
          parallel_grid_time = time.time() - start
          print('Parallel grid time (s): ', parallel_grid_time)
          
          vis_grid = grids_and_sum_weights[:, :, :, :, 0].todense()
          sum_weight = grids_and_sum_weights[:, :, 0, 0, 1].todense()
      else:
          #Memory problems if cube image is larger than memory. 
          grids_and_sum_weights = (grids_and_sum_weights.sum(axis=(0,1,3)))
          grids_and_sum_weights = grids_and_sum_weights.reshape(1,-1,*grids_and_sum_weights.shape[2:]) #reshape is needed to combine the gridded channels and the blockwise channels
          grids_and_sum_weights = grids_and_sum_weights.compute()
          
          parallel_grid_time = time.time() - start
          print('Parallel grid time (s): ', parallel_grid_time)
          
          
          vis_grid = grids_and_sum_weights[0,:, :, :, :, 0].todense()
          sum_weight = grids_and_sum_weights[0,:, :, 0, 0, 1].todense()
          
          
      total_grid_time = time.time() - start
      print('Total grid time (s): ', total_grid_time)
      
      n_pol = vis_grid.shape[1]
      
      grid_dict = {}
      coords = dict(zip(['chan', 'pol','u','v'], [list(range(ss)) for ss in vis_grid.shape]))
      
      grid_dict['VIS_GRID'] = xr.DataArray(da.from_array(vis_grid, chunks = (1,n_pol,grid_parms['imsize'][0],grid_parms['imsize'][1])), dims=['chan','pol','u', 'v'])
      grid_dict['SUM_WEIGHT'] = xr.DataArray(da.from_array(sum_weight, chunks = (1,n_pol)), dims=['chan','pol'])
      grid_dict['CORRECTING_CGK'] = xr.DataArray(da.from_array(correcting_cgk_image, chunks = (grid_parms['imsize'][0],grid_parms['imsize'][1])), dims=['u','v'])
      
      grid_dataset = xr.Dataset(grid_dict, coords=coords)
      grid_dataset.attrs['parallel_grid_time'] = parallel_grid_time
      grid_dataset.attrs['total_grid_time'] = total_grid_time
      
      return grid_dataset
      
      
      
      
      
      
      
