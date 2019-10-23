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

###############
def setmeta(df, fields={}):
    """
    .. todo::
        This function is not yet implemented

    Set the metadata contents of an MS Dataframe
    
    Parameters
    ----------
    df : Dask Dataframe
        input MS to view
    fields : dict
        dictionary of column:value pairs to set in the metadata

    Returns
    -------
    Dask Dataframe
        New Dataframe with modified metadata
    """
    return {}


#############
def viewmeta(df):
    """
    .. todo::
        This function is not yet implemented

    View the metadata of an MS Dataframe
    
    Parameters
    ----------
    df : Dask Dataframe
        input MS to view

    Returns
    -------
    dict
        Summary information
    """
    return {}


#############
def summarizeDF(df, field=None, spw=None, timerange=None, uvrange=None, antenna=None, scan=None):
    """
    .. todo::
        This function is not yet implemented

    Summarize the contents of an MS Dataframe
    
    Parameters
    ----------
    df : Dask Dataframe
        input MS to summarize
    field : int
        field selection. If None, use all fields
    spw : int
        spw selection. If None, use all spws
    timerange : int
        time selection. If None, use all times
    uvrange : int
        uvrange selection. If None, use all uvranges
    antenna : int
        antenna selection. If None, use all antennas
    scan : int
        scan selection. If None, use all scans

    Returns
    -------
    dict
        Summary information
    """
    return {}


###################
def summarizeFile(infile, ddi=None):
    """
    .. todo::
        This function is not yet implemented

    Summarize the contents of an Apache Parquet MS
    
    Parameters
    ----------
    infile : Dask Dataframe
        input filename of Parquet MS 
    ddi : int
        Data Description ID in MS to inspect. Defaults to None, which will summarize all

    Returns
    -------
    dict
        Summary information
    """
    return {}
