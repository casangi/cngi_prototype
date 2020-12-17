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
