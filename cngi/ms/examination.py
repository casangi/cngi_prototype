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

import os
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

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
def summarizeFile(infile, ddis=None):
    """
    Summarize the contents of an Apache Parquet MS
    
    Parameters
    ----------
    infile : Dask Dataframe
        input filename of Parquet MS 
    ddis : int or list
        Data Description IDs in MS to inspect. Default = None, summarizes all
    
    Returns
    -------
    Pandas Dataframe
        Summary information
    """
    
    if ddis == None:
        ddis = list(np.array(os.listdir(infile), dtype=int))
    elif type(ddis) != list:
        ddis = [ddis]
    
    summary = pd.DataFrame([])
    for ddi in ddis:
      dpath = os.path.join(infile, str(ddi))
      chunks = os.listdir(dpath)
      dsize = np.sum([os.path.getsize(os.path.join(dpath,ff)) for ff in chunks])/1024**3
      pqf = pq.ParquetFile(os.path.join(dpath,chunks[0]))
      sdf = [{'ddi':ddi, 'chunks':len(chunks), 'size_GB':np.around(dsize,2), 
              'row_count_estimate':pqf.metadata.num_rows*len(chunks),
              'col_count':pqf.metadata.num_columns, 'col_names':pqf.schema.names}]
      summary = pd.concat([summary,pd.DataFrame(sdf)], axis=0, sort=False)
    
    summary = summary.reset_index().drop(columns=['index'])
    return summary
