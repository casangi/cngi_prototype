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


def preview(xds, variable='image', region=None, pol=0, channels=0, tsize=250):
    """
    Preview the selected image component
    
    .. todo::
        Convert default axis display from radians to hourr:minute:second format

    Parameters
    ----------
    xds : xarray.core.dataset.Dataset
        input Image
    variable : str
        dataset variable to plot.  Default is image
    region : str
        dataset variable to use as a region/mask.
    pol : int
        polarization dimension index to plot.  Default is 0
    channels : int or list
        channel dimension index or indices to plot.  Default is 0
    tsize : int
        target size of the preview image (might be smaller). Default is 250 pixels
        
    Returns
    -------
      Open matplotlib window
    """
    import matplotlib.pyplot as plt
    from matplotlib import colors
    import numpy as np
    # try:
    #     import xrscipy
    # except ImportError as e:
    #     print(e)

    #plt.clf()
    channels = np.atleast_1d(channels)
    rows, cols = [int(np.ceil(len(channels)/4.0)), min(len(channels),4)]
    fig, axes = plt.subplots(rows, cols, 
                             figsize=(4*cols, 3*rows), 
                             constrained_layout=True)
    axes = np.atleast_2d(axes)
    
    # heuristic to determine thinning factor
    thinf = int(np.max(np.array(np.ceil(np.array(xds[variable].shape[:2]) / tsize), dtype=int)))
    
    # fast decimate to roughly the desired size
    if region is None:
        txds = xds[variable][{'chan':channels}].thin({'d0':thinf,
                                                           'd1':thinf,
                                                           'pol':1,
                                                           'chan':1})
    else:
        txds = ((xds[variable][{'chan':channels}] *
                 xds[region][{'chan':channels}]).thin({'d0':thinf,
                                                            'd1':thinf,
                                                            'pol':1,
                                                            'chan':1}))
    
    vmin, vmax = txds.values.min(), txds.values.max()
    xx, yy = 'd0', 'd1'
    if 'right_ascension' in txds.coords:
        xx, yy = 'right_ascension', 'declination'
    
    for ii,ch in enumerate(channels):
      # plot as a colormesh
      ixds = txds[dict(pol=pol, chan=ii)]
      im = ixds.plot.pcolormesh(ax=axes[ii//4,ii%4], x=xx, y=yy, add_colorbar=False,
                                vmin=vmin, vmax=vmax, norm=colors.PowerNorm(1))
      axes[ii//4,ii%4].set_title(variable + ' (' + str(pol) + ', ' + str(ch) +')')
      axes[ii//4,ii%4].invert_xaxis()
    
    fig.colorbar(im, ax=axes, shrink=0.6)
    
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
