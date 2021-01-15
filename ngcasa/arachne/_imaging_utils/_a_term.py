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

#ducting - code is complex and might fail after some time if parameters is wrong (time waisting). Sensable values are also checked. Gives printout of all wrong parameters. Dirty images alone has x parametrs.

import numpy as np

#@jit(nopython=True,cache=True)
def _outer_product(B1,B2):
    '''
    Input
    B1 2 x 2 x m x n array
    B2 2 x 2 x m x n array
    Output
    M 4 x 4 x m x n
    '''
    
    #assert B1.shape==B2.shape
    
    s = B1.shape
    
    M = np.zeros((4,4,s[2],s[3]),dtype=np.complex128)
    
    indx_b1 = np.array([[[0,0],[0,0],[0,1],[0,1]],[[0,0],[0,0],[0,1],[0,1]],[[1,0],[1,0],[1,1],[1,1]],[[1,0],[1,0],[1,1],[1,1]]])
    indx_b2 = np.array([[[0,0],[0,1],[0,0],[0,1]],[[1,0],[1,1],[1,0],[1,1]],[[0,0],[0,1],[0,0],[0,1]],[[1,0],[1,1],[1,0],[1,1]]])
    #print(indx_b1.shape)
    
    
    for i in range(4):
        for j in range(4):
            #print(indx_b1[i,j,:], ',*,', indx_b2[i,j,:])
            M[i,j,:,:] = B1[indx_b1[i,j,0],indx_b1[i,j,1],:,:] * B2[indx_b2[i,j,0],indx_b2[i,j,1],:,:]
    
    #print(M.shape)
    return(M)

def _outer_product_conv(B1,B2):
    '''
    Input
    B1 2 x 2 x m x n array
    B2 2 x 2 x m x n array
    Output
    M 4 x 4 x m x n
    '''
    #assert B1.shape==B2.shape
    
    s = B1.shape
    
    M = np.zeros((4,4,s[2],s[3]),dtype=np.complex128)
    
    indx_b1 = np.array([[[0,0],[0,0],[0,1],[0,1]],[[0,0],[0,0],[0,1],[0,1]],[[1,0],[1,0],[1,1],[1,1]],[[1,0],[1,0],[1,1],[1,1]]])
    indx_b2 = np.array([[[0,0],[0,1],[0,0],[0,1]],[[1,0],[1,1],[1,0],[1,1]],[[0,0],[0,1],[0,0],[0,1]],[[1,0],[1,1],[1,0],[1,1]]])
    
    for i in range(4):
        for j in range(4):
            M[i,j,:,:] = signal.fftconvolve(B1[indx_b1[i,j,0],indx_b1[i,j,1],:,:], B2[indx_b2[i,j,0],indx_b2[i,j,1],:,:],mode='same')
    
    print(M.shape)
    return(M)
    
def _make_flat(B):
    '''
    B 2x2xmxn
    B_flat 2mx2n
    '''
    s = B.shape
    B_flat = np.zeros((s[2]*s[0],s[3]*s[1]),dtype=complex)
    
    
    for i in range(s[0]):
        for j in range(s[1]):
            i_start = i*s[2]
            i_end = (i+1)*s[3]
            j_start = j*s[2]
            j_end = (j+1)*s[3]
            B_flat[i_start:i_end,j_start:j_end] = B[i,j,:,:]
            print(B[i,j,1024,1024],np.abs(B[i,j,1024,1024]))
    return B_flat
    
    
def _make_flat_casa(B):
    '''
    B mxnx16
    B_flat 4mx4n
    '''
    s = B.shape
    B_flat = np.zeros((s[0]*4,s[1]*4),dtype=complex)
    
    #indx = np.array([[0,0],[1,0],[2,0],[3,0],[0,1],[1,1],[2,1],[3,1],[0,2],[1,2],[2,2],[3,2],[0,3],[1,3],[2,3],[3,3]])
    indx = np.array([[0,0],[0,1],[0,2],[0,3],[1,0],[1,1],[1,2],[1,3],[2,0],[2,1],[2,2],[2,3],[3,0],[3,1],[3,2],[3,3]]) #saved as rows
    
    for c,i in enumerate(indx):
        #print(c,i)
        i_start = i[0]*s[0]
        i_end = (i[0]+1)*s[0]
        j_start = i[1]*s[1]
        j_end = (i[1]+1)*s[1]
        B_flat[i_start:i_end,j_start:j_end] = B[:,:,c].T
        print(B[1024,1024,c],np.abs(B[1024,1024,c]))
    return B_flat
    
def _compute_zc_coords(apeture_parms):
    print(apeture_parms)
    
    image_size = apeture_parms['zernike_size']
    image_center = image_size//2
    cell = 2./image_size #zernike polynomials defined on unit circle
    
    x = np.arange(-image_center[0], image_size[0]-image_center[0])*cell[0]
    y = np.arange(-image_center[1], image_size[1]-image_center[1])*cell[1]
    xy = np.array([x,y]).T
    x_grid, y_grid = np.meshgrid(x,y,indexing='ij')
    
    #print('image_size,cell',image_size,cell)
    #print('*******')
    #print(x_grid)
    #print(y_grid)
    #print('*******')
    
    parallactic_angle = - apeture_parms['parallactic_angle'] #- np.pi #clarify why + np.pi is necessary
    
    
    if parallactic_angle != 0:
        rot_mat = np.array([[np.cos(parallactic_angle),-np.sin(parallactic_angle)],[np.sin(parallactic_angle),np.cos(parallactic_angle)]]) #anti clockwise
        
        #r = np.einsum('ji, mni -> jmn', rot_mat, np.dstack([x_grid, y_grid]))
        '''
        x_grid_rot = np.cos(parallactic_angle)*x_grid - np.sin(parallactic_angle)*y_grid
        y_grid_rot = np.sin(parallactic_angle)*x_grid + np.cos(parallactic_angle)*y_grid
        '''
        x_grid_rot = np.cos(parallactic_angle)*x_grid + np.sin(parallactic_angle)*y_grid
        y_grid_rot = - np.sin(parallactic_angle)*x_grid + np.cos(parallactic_angle)*y_grid
        
        x_grid = x_grid_rot
        y_grid = y_grid_rot
    
    return x_grid, y_grid


def _create_chan_map(mxds,gcf_parms,sel_parms):

    vis_dataset = mxds.attrs[sel_parms['xds']]
    freq_chan = vis_dataset.chan.data
    chan_tolerance_factor = gcf_parms['chan_tolerance_factor']

    print(freq_chan)
    n_chan = len(freq_chan)
    cf_chan_map = np.zeros((n_chan,),dtype=int)
    
    orig_width = (np.max(freq_chan) - np.min(freq_chan))/len(freq_chan)
    
    tol = np.max(freq_chan)*chan_tolerance_factor
    n_pb_chan = int(np.floor( (np.max(freq_chan)-np.min(freq_chan))/tol) + 0.5) ;

    #Create PB's for each channel
    if n_pb_chan == 0:
        n_pb_chan = 1
    
    if n_pb_chan >= n_chan:
        cf_chan_map = np.arange(n_chan)
        pb_freq = freq_chan
        return cf_chan_map, pb_freq
    
    pb_delta_bandwdith = (np.max(freq_chan) - np.min(freq_chan))/n_pb_chan
    pb_freq = np.arange(n_pb_chan)*pb_delta_bandwdith + np.min(freq_chan) + pb_delta_bandwdith/2

    cf_chan_map = np.zeros((n_chan,),dtype=int)
    for i in range(n_chan):
        cf_chan_map[i],_ = _find_nearest(pb_freq, freq_chan[i])

    

    return cf_chan_map, pb_freq
    
def _find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx, array[idx]


##########################
def _create_beam_map(mxds,sel_parms):
    import xarray as xr
    import dask.array as da

    beam_ids = mxds.beam_ids.data.compute()
    n_beam = len(beam_ids)
    feed_beam_ids = mxds.FEED.beam_id.data.compute()
    
    #Assuming antenna ids remain constant over time and that feed and antenna ids line up
    #ant1_id = mxds.attrs[sel_parms['xds']].ANTENNA1[0,:]
    #ant2_id = mxds.attrs[sel_parms['xds']].ANTENNA2[0,:]
    ant1_id = mxds.attrs[sel_parms['xds']].ANTENNA1.data.compute()
    ant2_id = mxds.attrs[sel_parms['xds']].ANTENNA2.data.compute()
    
    unified_ant_indx = np.zeros((ant1_id.shape[1],2),dtype=int)
    for i in range(ant1_id.shape[1]): #for baseline
        id1 = np.unique(ant1_id[:,i])
        id1 = id1[id1 >= 0]
        id2 = np.unique(ant2_id[:,i])
        id2 = id2[id2 >= 0]
        #unified_ant_indx
        assert(len(id1) == 1 and len(id2) == 1), "Antennas not consistant over time."
        
        unified_ant_indx[i,:] = [id1[0],id2[0]]
        
    baseline_beam_id = np.full((len(unified_ant_indx),2), np.nan, dtype=np.int32)
    ant_ids = mxds.antenna_ids #same length as feed_beam_ids
    
    for indx,id in enumerate(ant_ids.data):
        baseline_beam_id[unified_ant_indx[:,0]==id,0] = feed_beam_ids[indx]
        baseline_beam_id[unified_ant_indx[:,1]==id,1] = feed_beam_ids[indx]
        

    beam_pair_id = find_unique_pairs(baseline_beam_id)
    #n_beam_pair = #max possible is int((n_beam**2 + n_beam)/2)

    beam_map = np.zeros((len(unified_ant_indx),),dtype=int)

    for k,ij in enumerate(beam_pair_id):
        #print(k,ij)
        beam_map[(baseline_beam_id[:,0] == ij[0]) & (baseline_beam_id[:,1] == ij[1])] = k
        beam_map[(baseline_beam_id[:,1] == ij[0]) & (baseline_beam_id[:,0] == ij[1])] = k
        
    vis_dataset = mxds.attrs[sel_parms['xds']]
    baseline_chunksize = vis_dataset[sel_parms['data']].chunks[1][0]
    
    beam_map = xr.DataArray(da.from_array(beam_map,chunks=(baseline_chunksize)), dims=('baseline'))
    beam_pair_id = xr.DataArray(da.from_array(beam_pair_id,chunks=beam_pair_id.shape), dims=('beam_pair','pair'))
    return beam_map,beam_pair_id

    
    
    '''
    ant_ids = mxds.antenna_ids
    
    
    
    model_id = mxds.ANTENNA['MODEL'].data.compute()
    unique_model_id = mxds.ANTENNA['model_id'].data.compute()
    n_unique_model = len(unique_model_id)
    n_unique_model_pairs = int((n_unique_model**2 + n_unique_model)/2)
    
    #Assuming antenna ids remain constant over time
    ant1_id = mxds.attrs[sel_parms['xds']].ANTENNA1[0,:]
    ant2_id = mxds.attrs[sel_parms['xds']].ANTENNA2[0,:]
    
    baseline_model_indx = np.zeros((len(ant1_id),2),dtype=int)
    #print(baseline_model_indx.shape)
    
    #print(ant1_id.values)
    #print(ant2_id.values)
    
    
    for indx,id in enumerate(ant_ids):
        baseline_model_indx[ant1_id==id,0] = model_id[indx]
        baseline_model_indx[ant2_id==id,1] = model_id[indx]
        
    #print(baseline_model_indx)
        
    pb_ant_pairs = np.zeros((n_unique_model_pairs,2),dtype=int)
    k = 0
    for i in range(n_unique_model):
        for j in range(i,n_unique_model):
           pb_ant_pairs[k,:] = [unique_model_id[i],unique_model_id[j]]
           k = k + 1
    
    cf_baseline_map = np.zeros((len(ant1_id),),dtype=int)
    #print(cf_baseline_map.shape)
    
    for k,ij in enumerate(pb_ant_pairs):
        #print(k,ij)
        cf_baseline_map[(baseline_model_indx[:,0] == ij[0]) & (baseline_model_indx[:,1] == ij[1])] = k
        cf_baseline_map[(baseline_model_indx[:,1] == ij[0]) & (baseline_model_indx[:,0] == ij[1])] = k
    
    return cf_baseline_map,pb_ant_pairs
    '''


#Order does not matter
#https://stackoverflow.com/questions/47531532/find-unique-pairs-of-array-with-python
def find_unique_pairs(a):
    s = np.max(a)+1
    p1 = a[:,1]*s + a[:,0]
    p2 = a[:,0]*s + a[:,1]
    p = np.maximum(p1,p2)
    sidx = p.argsort(kind='mergesort')
    ps = p[sidx]
    m = np.concatenate(([True],ps[1:] != ps[:-1]))
    sm = sidx[m]
    #print(m)
    #print(a[sm,:])
    
    return a[sm,:]
    
    #b = np.array(list(set(tuple(sorted([m, n])) for m, n in zip(a[:,0], a[:,1]))))
    #print(b)

    '''
    from collections import Counter
    ctr = Counter(frozenset(x) for x in a)
    b = [ctr[frozenset(x)] == 1 for x in a]
    
    print(a)
    print(b)
    print(a[b,:])

    
    a = np.array(a)
    a.sort(axis=1)
    b = np.ascontiguousarray(a).view(
            np.dtype((np.void, a.dtype.itemsize * a.shape[1]))
            )
    q, inv, ct = np.unique(b, return_inverse=True, return_counts=True)
    print(q,inv,ct,ct[inv] == 1)
    '''
