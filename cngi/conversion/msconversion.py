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


#########################################
def read_legacy_ms(infile, ddi=None):
    """
    .. todo::
        This function is not yet implemented

    Read legacy CASA MS format data directly to an xarray Visibility Dataset

    Parameters
    ----------
    infile : str
        Input MS filename
    ddi : int
        Data Description ID in MS to read. If None, defaults to 0

    Returns
    -------
    xarray Dataset
        New Visibility Dataset of MS contents
    """
    return True



###########################################
def zarr_to_ms(infile, format='ms', outfile=None):
    """
    .. todo::
        This function is not yet implemented

    Convert xarray Visibility Dataset from zarr format to Legacy CASA MS or FITS format

    Parameters
    ----------
    infile : str
        Input zarr filename
    format : str
        Conversion output format, 'ms' or 'fits'.  Default = 'ms'
    outfile : str
        Output MS filename. If None, will use infile name with .ms extension

    Returns
    -------
    bool
        Success status
    """
    return True



###########################################
def asdm_to_zarr(infile, outfile=None):
    """
    .. todo::
        This function is not yet implemented

    Convert ASDM format to xarray Visibility Dataset zarr format (future)
    
    Parameters
    ----------
    infile : str
        Input ASDM filename
    outfile : str
        Output zarr filename. If None, will use infile name with .zarr extension

    Returns
    -------
    bool
        Success status
    """
    return True



###########################################
def zarr_to_asdm(infile, outfile=None):
    """
    .. todo::
        This function is not yet implemented

    Convert xarray Visibility Dataset from zarr format to ASDM format (future)

    Parameters
    ----------
    infile : str
        Input zarr filename
    outfile : str
        Output ASDM filename. If None, will use infile name with .asdm extension

    Returns
    -------
    bool
        Success status
    """
    return True
