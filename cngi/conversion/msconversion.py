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

import pandas as pd
import os
import numpy as np


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



##########################################
def ms_to_pq(infile, outfile=None, ddi=None, membudget=500e6, maxchunksize=1000000):
    """
    Convert legacy format MS to Apache Parquet format MS

    This function requires CASA6 casatools module. 

    Parameters
    ----------
    infile : str
        Input MS filename
    outfile : str
        Output Parquet filename. If None, will use infile name with .pq extension
    ddi : int
        specific ddi to convert. Leave as None to convert entire MS
    membudget : float
        Target in-memory byte size of a chunk, Default = 500e6 (~= 500MB)
    maxchunksize : int
        Maximum number of rows allowed per chunk        

    Returns
    -------
    """    
    from casatools import table as tb
    from cngi.direct import GetFrameworkClient
    import pyarrow as pa
    import pyarrow.parquet as pq
        
    # parse filename to use
    infile = os.path.expanduser(infile)
    prefix = infile[:infile.rindex('.')]
    if outfile == None: outfile = prefix + '.pq'
    
    # need to manually remove existing parquet file (if any)
    tmp = os.system("rm -fr " + outfile)
    tmp = os.system("mkdir " + outfile)
    
    MS = tb(infile)
    MS.open(infile, nomodify=True, lockoptions={'option':'usernoread'})
    
    # let's assume that each DATA_DESC_ID is a fixed shape that may differ from others
    # process each DATA_DESC_ID and place it in its own partition
    ddis = MS.taql('select distinct DATA_DESC_ID from %s' % prefix+'.ms').getcol('DATA_DESC_ID')
    
    MS.close()
    
    #######################
    # process a DDI from the input MS, assume a fixed shape within the ddi (should always be true)
    # each DDI is written to its own subdirectory under the parent parquet folder
    # consequently, different DDI's may be processed in parallel if the MS is opened with no locks
    def processDDI(ddi, infile, outfile, membudget, maxchunksize):
      MS = tb(infile)
      MS.open(infile, nomodify=True, lockoptions={'option':'usernoread'})  # allow concurrent reads
      MSDDI = MS.taql('select * from %s where DATA_DESC_ID = %s' % (infile,str(ddi))) # subselect this ddi
      nrows = MSDDI.nrows()
      cols = MS.colnames()
      
      # compute the size of first row to estimate the number or rows that will fit in mem
      col_count, channels = 0, 0
      for col in cols:
        col_count += 1
        if MSDDI.isvarcol(col):
          try:
            col_shape = np.array(eval(MSDDI.getcolshapestring(col,nrow=1)[0]))
            if col_shape[0] != channels: col_count += col_shape[0]-1
            if (len(col_shape) > 1) and (channels == 0): channels = col_shape[1]
            if MSDDI.coldatatype(col) == 'complex': col_count += 1
          except Exception:  # sometimes bad columns break the table tool (??)
            cols = [_ for _ in cols if _ != col]
      
      # adjust chunksize to fit in memory
      chunksize = np.min((maxchunksize, int(membudget/(col_count*channels*8))))
      
      # process MS data_desc_id in chunks of computed size
      for cc,rr in enumerate(range(0,nrows,chunksize)):
        chunk = np.arange(min(chunksize,nrows-rr))+rr
        if cc==0: print('processing ddi %s: chunks=%s, size=%s' % (str(ddi),str(nrows//chunksize),str(chunksize)))
        
        # build python dictionary one MS column at a time
        mdi, chancount = {}, 1
        mdi['SAMPLE'] = chunk
        for col in cols:
          if col == 'DATA_DESC_ID': continue
          try: # every column should be a fixed size within a given ddi
            marr = MSDDI.getcol(col, rr, len(chunk))
            if col == 'UVW':
              mdi['U'], mdi['V'], mdi['W'] = marr[0,:], marr[1,:], marr[2,:]
            elif marr.ndim == 1:
              mdi[col] = marr
            elif marr.ndim == 2:
              if (chancount > 1) and (len(marr) == chancount):
                mdi[col] = marr.reshape(-1)
              else:
                mdi.update(dict(zip([col+str(ii) for ii in range(marr.shape[0])], marr)))
            elif marr.ndim == 3: 
              mdi['CHAN'] = np.repeat(np.arange(marr.shape[1]), marr.shape[-1])
              chancount = marr.shape[1]
              marr = marr.reshape(marr.shape[0], -1)
              ncs = [col+str(ii) for ii in range(marr.shape[0])]
              if marr.dtype == 'complex128':
                mdi.update(dict(zip(['R'+cc for cc in ncs], np.real(marr))))
                mdi.update(dict(zip(['I'+cc for cc in ncs], np.imag(marr))))
              else:
                mdi.update(dict(zip(ncs, marr)))
            else: #ndim > 3
              print("WARNING: can't process shape of column %s"%(col))
              cols = [_ for _ in cols if _ != col]
          except Exception:  # sometimes bad columns break the table tool (??)
            print("WARNING: can't process column %s"%(col))
            cols = [_ for _ in cols if _ != col]
        
        for key in mdi.keys():
          if len(mdi[key]) == len(chunk):
            mdi[key] = np.tile(mdi[key], chancount)
        
        # write to your favorite ots format
        pq.write_to_dataset(pa.Table.from_pydict(mdi), root_path=outfile+'/'+str(ddi),
                            version='2.0', compression='ZSTD')
      
      MS.close()
      print("completed ddi " + str(ddi))
    #############################
    
    # parallelize with direct interface
    client = GetFrameworkClient()
    if ddi != None:
        processDDI(ddi,infile,outfile,membudget,maxchunksize)
    elif client == None:
        for ddi in ddis:
            processDDI(ddi,infile,outfile,membudget,maxchunksize)
    else:
        jobs = client.map(processDDI, ddis, 
                          np.repeat(infile, len(ddis)), 
                          np.repeat(outfile, len(ddis)), 
                          np.repeat(membudget, len(ddis)),
                          np.repeat(maxchunksize, len(ddis)))
        
        # block until complete
        for job in jobs: job.result()
    print('Complete.')





##########################################
def ms_to_zarr(infile, outfile=None, ddi=None, membudget=500e6, maxchunksize=1000):
    """
    Convert legacy format MS to a zarr xarray Dataset format MS 

    This function requires CASA6 casatools module. 

    Parameters
    ----------
    infile : str
        Input MS filename
    outfile : str
        Output zarr filename. If None, will use infile name with .zarr extension
    ddi : int
        specific ddi to convert. Leave as None to convert entire MS
    membudget : float
        Target in-memory byte size of a chunk, Default = 500e6 (~= 500MB)
    maxchunksize : int
        Maximum number of rows allowed per chunk        

    Returns
    -------
    """    
    from itertools import cycle
    from casatools import table as tb
    from xarray import Dataset as xd
    from xarray import DataArray as xa
    from numcodecs import Blosc
    
    # parse filename to use
    infile = os.path.expanduser(infile)
    prefix = infile[:infile.rindex('.')]
    if outfile == None: outfile = prefix + '.zarr'
    
    # need to manually remove existing directory (if any)
    tmp = os.system("rm -fr " + outfile)
    tmp = os.system("mkdir " + outfile)
    
    MS = tb(infile)
    MS.open(infile, nomodify=True, lockoptions={'option':'usernoread'})
    
    # let's assume that each DATA_DESC_ID is a fixed shape that may differ from others
    # process each DATA_DESC_ID and place it in its own partition
    ddis = MS.taql('select distinct DATA_DESC_ID from %s' % prefix+'.ms').getcol('DATA_DESC_ID')
    
    MS.close()
    
    #######################
    # process a DDI from the input MS, assume a fixed shape within the ddi (should always be true)
    # each DDI is written to its own subdirectory under the parent folder
    # consequently, different DDI's may be processed in parallel if the MS is opened with no locks
    def processDDI(ddi, infile, outfile, membudget, maxchunksize):
      tmp = os.system("mkdir " + outfile + '/' + str(ddi))
      MS = tb(infile)
      MS.open(infile, nomodify=True, lockoptions={'option':'usernoread'})  # allow concurrent reads
      MSDDI = MS.taql('select * from %s where DATA_DESC_ID = %s' % (infile,str(ddi))) # subselect this ddi
      nrows = MSDDI.nrows()
      cols = MS.colnames()
      compressor = Blosc(cname='zstd', clevel=2, shuffle=0)
      
      # compute the size of first row to estimate the number or rows that will fit in mem
      row_size = 0
      for col in cols:
        row_size += 8
        if MSDDI.isvarcol(col):
          try:
            row_size += 8*np.array(eval(MSDDI.getcolshapestring(col,nrow=1)[0])).prod()-8
            if MSDDI.coldatatype(col) == 'complex': # twice the size
              row_size += 8*np.array(eval(MSDDI.getcolshapestring(col,nrow=1)[0])).prod()
          except Exception:  # sometimes bad columns break the table tool (??)
            cols = [_ for _ in cols if _ != col]
      
      # adjust chunksize to fit in memory
      chunksize = np.min((maxchunksize, int(membudget/row_size)))
      
      # process MS data_desc_id in chunks of computed size
      for cc,rr in enumerate(range(0,nrows,chunksize)):
        chunk = np.arange(min(chunksize,nrows-rr))+rr
        if cc==0: print('processing ddi %s: chunks=%s, size=%s' % (str(ddi),str(nrows//chunksize),str(chunksize)))
        
        # build python dictionary one MS column at a time
        coords, xdas = {'rows':chunk}, {}
        for col in cols:
          if col=='DATA_DESC_ID': continue
          try: # every column should be a fixed size within a given ddi
            marr = MSDDI.getcol(col, rr, len(chunk)).transpose()
            if marr.dtype == 'bool': marr = marr.astype(int)
            if col == 'UVW': 
              xdas[col] = xa(marr, dims=['rows','uvw'])
              coords['uvw'] = np.arange(3)
            elif marr.ndim == 1:
              coords[col] = ('rows', marr)
            elif marr.ndim == 2:  # determine if (pol x row) or (chan x row)
              if col in ['WEIGHT', 'SIGMA']:
                coords['pols'] = np.arange(marr.shape[-1])
              if ('pols' in coords) and (marr.shape[-1] == coords['pols'].shape[0]):
                xdas[col] = xa(marr, dims=['rows','pols'])
              elif ('chans' in coords) and (marr.shape[-1] == coords['chans'].shape[0]):
                xdas[col] = xa(marr, dims=['rows','chans'])
              else:
                print("WARNING: unknown dimension in column %s"%(col))
                cols = [_ for _ in cols if _ != col]
            elif marr.ndim == 3:
              xdas[col] = xa(marr, dims=['rows','chans','pols'])
              coords['chans'] = np.arange(marr.shape[1])
            else: # ndim > 3
              print("WARNING: can't process shape of column %s"%(col))
              cols = [_ for _ in cols if _ != col]
          except Exception:  # sometimes bad columns break the table tool (??)
            print("WARNING: can't process column %s"%(col))
            cols = [_ for _ in cols if _ != col]
        
        xds = xd(xdas, coords=coords).chunk({'rows':len(coords['rows']),'chans':None,'pols':None,'uvw':None})
        if cc==0: 
            encoding = dict(zip(list(xds.data_vars), cycle([{'compressor':compressor}])))
            xds.to_zarr(outfile+'/'+str(ddi), mode='w', encoding=encoding)
        else:
            xds.to_zarr(outfile+'/'+str(ddi), mode='a', append_dim='rows')
      
      MS.close()
      print("completed ddi " + str(ddi))
    #############################
    
    if ddi != None:
        processDDI(ddi,infile,outfile,membudget,maxchunksize)
    else:
        for ddi in ddis:
            processDDI(ddi,infile,outfile,membudget,maxchunksize)



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
