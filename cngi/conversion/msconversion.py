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

    Read legacy CASA MS format data directly to a Dask Dataframe

    Parameters
    ----------
    infile : str
        Input MS filename
    ddi : int
        Data Description ID in MS to read. If None, defaults to 0

    Returns
    -------
    Dask Dataframe
        New Dataframe of MS contents
    """
    return True



###########################################
def pq_to_ms(infile, outfile=None):
    """
    .. todo::
        This function is not yet implemented

    Convert Apache Parquet MS to Legacy CASA MS format

    Parameters
    ----------
    infile : str
        Input Parquet filename
    outfile : str
        Output MS filename. If None, will use infile name with .ms extension

    Returns
    -------
    bool
        Success status
    """
    return True



###########################################
def asdm_to_pq(infile, outfile=None):
    """
    .. todo::
        This function is not yet implemented

    Convert ASDM format to Apache Parquet format (future)
    
    Parameters
    ----------
    infile : str
        Input ASDM filename
    outfile : str
        Output Parquet filename. If None, will use infile name with .pq extension

    Returns
    -------
    bool
        Success status
    """
    return True



###########################################
def pq_to_asdm(infile, outfile=None):
    """
    .. todo::
        This function is not yet implemented

    Convert Apache Parquet MS to ASDM format (future)
    
    Parameters
    ----------
    infile : str
        Input Parquet filename
    outfile : str
        Output ASDM filename. If None, will use infile name with .asdm extension

    Returns
    -------
    bool
        Success status
    """
    return True



###########################################
def fits_to_pq(infile, outfile=None):
    """
    .. todo::
        This function is not yet implemented

    Convert FITS format MS to Apache Parquet format (future)
    
    Parameters
    ----------
    infile : str
        Input FITS filename
    outfile : str
        Output Parquet filename. If None, will use infile name with .pq extension

    Returns
    -------
    bool
        Success status
    """
    return True



############################################
def pq_to_fits(infile, outfile=None):
    """
    .. todo::
        This function is not yet implemented

    Convert Apache Parquet MS to FITS format (future)
    
    Parameters
    ----------
    infile : str
        Input Parquet filename
    outfile : str
        Output FITS filename. If None, will use infile name with .fits extension

    Returns
    -------
    bool
        Success status
    """
    return True
