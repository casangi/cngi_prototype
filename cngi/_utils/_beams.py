#  CASA Next Generation Infrastructure
#  Copyright (C) 2021 AUI, Inc. Washington DC, USA
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

#################################
# Helper File
#
# Not exposed in API
#
#################################


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
    return f_xy, np.array([alpha, beta, gamma])


