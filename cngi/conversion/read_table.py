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
"""
this module will be included in the api
"""


def read_table(infile, subtable=None, timecols=None, ignorecols=None):
    """
    Read generic casacore table format to xarray Dataset

    Parameters
    ----------
    infile : str
        Input table filename
    subtable : str
        Name of the subtable to process. If None, main table will be used
    timecols : list
        list of strings specifying column names to convert to datetime format from casacore time.  Default is None
    ignorecols : list
        list of column names to ignore. This is useful if a particular column is causing errors.  Default is None
    
    Returns
    -------
    New xarray.core.dataset.Dataset
      New xarray Dataset of table data contents. One element in list per DDI plus the metadata global.
    """
    import os
    import cngi._utils._table_conversion2 as tblconv
    
    if subtable is None: subtable = ''
    if timecols is None: timecols = []
    if ignorecols is None: ignorecols = []
    
    xds = tblconv.read_simple_table(os.path.expanduser(infile), subtable=subtable, timecols=timecols, ignore=ignorecols)
    return xds
