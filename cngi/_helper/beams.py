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


#####
## compute the synthesized beam for a grid of specified size
## equation drawn from: https://casa.nrao.edu/casadocs/latest/casa-fundamentals/definition_synthesized_beam
## d1, d2, incr in arcseconds, theta in degrees, incr in radians
def synthesizedbeam(d1, d2, theta, x_dim, y_dim, incr):
    import numpy as np
    import xarray

    incr = np.abs(incr) * 180 / np.pi * 60 * 60
    
    # make xy coordinates from grid size
    xx, yy = np.mgrid[:x_dim, :y_dim]
    xy = np.column_stack((xx.ravel() - x_dim//2, yy.ravel() - y_dim//2))
    
    # convert d1 and d2 from arcseconds to number of cells
    d1 = d1 / np.abs(incr[0])
    d2 = d2 / np.abs(incr[1])
    theta = theta * np.pi / 180
    
    alpha = 4 * np.log(2) * ( (np.cos(theta)**2 / d1**2) + (np.sin(theta)**2 / d2**2) )
    beta = 8 * np.log(2) * ( (1/d1**2) - (1/d2**2) ) * np.sin(theta) * np.cos(theta)
    gamma = 4 * np.log(2) * ( (np.sin(theta)**2 / d1**2) + (np.cos(theta)**2 / d2**2) )
    
    f_xy = np.exp(-((alpha * xy[:,0]**2) + (beta*xy[:,1]*xy[:,0]) + (gamma * xy[:,1]**2)))
    f_xy = f_xy.reshape(x_dim, y_dim)
    return f_xy


