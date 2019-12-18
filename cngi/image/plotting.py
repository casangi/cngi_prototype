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




########################
def preview(xds, variable='image', stokes=0, channel=0, tsize=250):
    """
    Preview the selected image component
    
    Parameters
    ----------
    xds : xarray Dataset
        input image dataset
    variable : str
        dataset variable to plot.  Default is image
    stokes : int
        stokes dimension index to plot.  Defualt is 0
    channel : int
        channel dimension index to plot.  Default is 0
    tsize : int
        target size of the preview image (might be smaller). Default is 250 pixels
        
    Returns
    -------
      Open matplotlib window
    """
    import matplotlib.pyplot as plt
    from matplotlib import colors
    import numpy as np
    
    plt.clf()
    
    # fast decimate to roughly the desired size
    thinfactor = np.max(np.array(np.ceil(np.array(xds[variable].shape[:2])/tsize), dtype=int))
    txds = xds[variable][dict(stokes=stokes, frequency=channel)].thin(int(thinfactor))
        
    # plot as a colormesh
    txds.plot.pcolormesh(x='right_ascension', y='declination', norm=colors.PowerNorm(1))
    plt.title('stokes = ' + str(stokes) + '   channel = ' + str(channel))
    plt.gca().invert_xaxis()
    
    # TODO
    # convert axes from radians to hr:minute:second format
    #ra_ticks = np.array(plt.gca().get_xticks().tolist())
    #ra_hrs = ra_ticks/(2*np.pi) * 24
    #ra_min = np.abs(ra_hrs) % 1 * 60
    #ra_sec = ra_min % 1 * 60
    #ra_str = ['{:02.0f}:{:02.0f}:{:06.3f}'.format(np.floor(ra_hrs[0]),np.floor(ra_min[0]),ra_sec[0])]
    #ra_str += ['{:06.3f}'.format(ra_sec[ii]) for ii in range(1,len(ra_hrs))]
    #plt.gca().set_xticklabels(ra_str)
    
    plt.show(block=False)
