#   Copyright 2020 AUI, Inc. Washington DC, USA
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

#################################
# Helper File
#
# Not exposed in API
#
#################################
import os
from casatools import table as tb
from numcodecs import Blosc
import pandas as pd
import xarray
import numpy as np
from itertools import cycle
import warnings
import time

warnings.filterwarnings('ignore', category=FutureWarning)



########################################################
# helper for reading time columns to datetime format
# pandas datetimes are referenced against a 0 of 1970-01-01
# CASA's modified julian day reference time is (of course) 1858-11-17
# this requires a correction of 3506716800 seconds which is hardcoded to save time
def convert_time(rawtimes):
    correction = 3506716800.0
    return pd.to_datetime(np.array(rawtimes) - correction, unit='s').values



######################################
# compute dimensions of variable shaped columns
# this will be used to standardize the shape to the largest value of each dimension
# this does not work on columns with variable dimension lengths (i.e. ndim changes between 2 and 3)
# columns designated as ignore will be treated as bad
def compute_dimensions(tbobj, ignore=[]):
    cshape = {}
    bad_cols = []
    for col in tbobj.colnames():
        if col in ignore: continue
        if not tbobj.iscelldefined(col, 0):
            bad_cols += [col]
            continue
        try:
            if tbobj.isvarcol(col):
                # find unique shapes in this col, if there is only 1, it isn't really a var col
                tshape = np.unique([np.unique(tbobj.getcolshapestring(col, sidx, 10000)) for sidx in range(0, tbobj.nrows(), 10000)])
                if len(tshape) == 1: continue
                tshape = [list(np.fromstring(nn[1:-1], dtype=int, sep=',')) for nn in tshape]  # list of int shapes
                if len(np.unique([len(nn) for nn in tshape])) > 1:
                    print('##### ERROR processing column %s, shape has variable dimensionality, skipping...' % col)
                    bad_cols += [col]
                    continue
                cshape[col] = np.max([nn for nn in tshape], axis=0)  # store the max dimensionality of this col
        except Exception:
            bad_cols += [col]
            continue
    return cshape, bad_cols



##################################################################
# convert a legacy casacore table format to CNGI xarray/zarr
# infile/outfile can be the main table or specific subtable
# if infile/outfile are the main table, subtable can also be specified
# dimnames is a list used to override default dimension names
# timecols is a list of column names to convert to datetimes
# ignore is a list of columns to ignore
def convert_simple_table(infile, outfile, subtable='', dimnames=None, timecols=[], ignore=[], compressor=None, chunks=(40000, 20, 1), nofile=False):
    
    if compressor is None:
        compressor = Blosc(cname='zstd', clevel=2, shuffle=0)
    
    tb_tool = tb()
    tb_tool.open(os.path.join(infile, subtable), nomodify=True, lockoptions={'option': 'usernoread'})  # allow concurrent reads
    
    # sometimes they are empty
    if tb_tool.nrows() == 0:
        tb_tool.close()
        return xarray.Dataset()
    
    # handle no chunking of first axis
    if chunks[0] == -1:
        chunks[0] = tb_tool.nrows()
    
    # master dataset holders
    mvars, mcoords = {}, {}
    mdims = {'d0': tb_tool.nrows()}  # keys are dimension names, values are dimension sizes
    cshape, bad_cols = compute_dimensions(tb_tool, ignore)
    
    for start_idx in range(0, tb_tool.nrows(), chunks[0]):
        for col in tb_tool.colnames():
            if (col in ignore) or (col in bad_cols): continue
            tblname = tb_tool.name().split('/')[-1]
            print('reading %s chunk %s of %s, col %s...%s' % (tblname, str(start_idx // chunks[0]), str(tb_tool.nrows() // chunks[0]), col, ' '*20), end='\r')
            try:
                if col in cshape:   # if this column has a varying shape, it needs to be normalized
                    data = tb_tool.getvarcol(col, start_idx, chunks[0])
                    tshape = cshape[col]  # grab the max dimensionality of this col
                    # expand the variable shaped column to the maximum size of each dimension
                    # blame the person who devised the tb.getvarcol return structure
                    data = np.array([np.pad(data['r' + str(kk)][..., 0], np.stack((tshape * 0, tshape - data['r' + str(kk)].shape[:-1]), -1)).transpose()
                                     for kk in np.arange(start_idx + 1, start_idx + len(data) + 1)])
                else:
                    data = tb_tool.getcol(col, start_idx, chunks[0]).transpose()
            except Exception:
                bad_cols += [col]
                continue
            
            # sometimes, even though the table has >0 rows, certain columns still return 0 rows
            if len(data) == 0: continue
            
            # convert col values to datetime if desired
            if col in timecols:
                data = convert_time(data)

            # if this column has additional dimensionality, we need to create/reuse placeholder names
            # then apply any specified names
            dims = ['d0']
            for ii, dd in enumerate(data.shape[1:]):
                if (ii+1 >= len(mdims)) or (dd not in list(mdims.values())[ii+1:]):
                    mdims['d%i' % len(mdims.keys())] = dd
                dims += [list(mdims.keys())[ii+1:][list(mdims.values())[ii+1:].index(dd)]]
            if dimnames is not None: dims = (dimnames[:len(dims)] + dims[len(dims)-len(dimnames)-1:])[:len(dims)]
            
            chunking = [chunks[di] if di < len(chunks) else chunks[-1] for di, dk in enumerate(dims)]
            chunking = [cs if cs > 0 else data.shape[ci] for ci, cs in enumerate(chunking)]
            
            # store ID columns as a list of coordinates, otherwise store as a list of data variables
            if col.upper().endswith('_ID'):
                mcoords[col.lower()] = xarray.DataArray(data, dims=dims)
            else:
                mvars[col.upper()] = xarray.DataArray(data, dims=dims).chunk(dict(zip(dims, chunking)))
            
        xds = xarray.Dataset(mvars, coords=mcoords)
        if (not nofile) and (start_idx == 0):
            encoding = dict(zip(list(xds.data_vars), cycle([{'compressor': compressor}])))
            if len(subtable) > 0: xds = xds.assign_attrs({'name': subtable+' table'})
            if len(bad_cols) > 0: xds = xds.assign_attrs({'bad_cols': bad_cols})
            xds.to_zarr(os.path.join(outfile,subtable), mode='w', encoding=encoding, consolidated=True)
        elif not nofile:
            xds.to_zarr(os.path.join(outfile,subtable), mode='a', append_dim='d0', compute=True, consolidated=True)
    
    tb_tool.close()
    if not nofile:
        xds = xarray.open_zarr(os.path.join(outfile,subtable))
    
    return xds



#keys = {'TIME': 'time', ('ANTENNA1', 'ANTENNA2'): 'baseline'}

#############################################################################
# convert a legacy casacore table format to CNGI xarray/zarr with dimension expansion from splitting apart rows by values in specified columns
# infile/outfile can be the main table or specific subtable
# if infile/outfile are the main table, subtable can also be specified
# keys are a dict mapping source columns to target dims, use a tuple when combining cols (ie {('ANTENNA1','ANTENNA2'):'baseline'}
# subsel is a dict of col name : col val to subselect in the table (ie {'DATA_DESC_ID' : 0}
# timecols is a list of column names to convert to datetimes
# dimnames is a dictionary mapping old to new dimension names for remaining dims (not in keys)
# ignore is a list of column names to ignore
def convert_expanded_table(infile, outfile, keys, subtable='', subsel=None, timecols=[], dimnames={}, ignore=[], compressor=None, chunks=(100, 20, 1), nofile=False):
    
    if compressor is None:
        compressor = Blosc(cname='zstd', clevel=2, shuffle=0)
    
    tb_tool = tb()
    tb_tool.open(os.path.join(infile,subtable), nomodify=True, lockoptions={'option': 'usernoread'})  # allow concurrent reads
    
    # sometimes they are empty
    if tb_tool.nrows() == 0:
        tb_tool.close()
        return xarray.Dataset()
        
    # handle no chunking of first axis
    if chunks[0] == -1:
        chunks[0] = tb_tool.nrows()
    
    # sort table by value of dimensions to be expanded
    # load and store the column to be used as the key for the first dimension (aka the row key)
    # compute the following:
    #  1. unique_row_keys = unique values of this key (size of first dim)
    #  2. row_changes = row number where the value changes
    #  3. row_idxs = row numbers where each unique value occurs
    # then compute 1 and 3 for each additional key/dimension and store in midxs dictionary
    ordering = ','.join([np.atleast_1d(key)[ii] for key in keys.keys() for ii in range(len(np.atleast_1d(key)))])
    if subsel is None:
        sorted_table = tb_tool.taql('select * from %s ORDERBY %s' % (os.path.join(infile,subtable), ordering))
    else:
        tsel = [list(subsel.keys())[0], list(subsel.values())[0]]
        sorted_table = tb_tool.taql('select * from %s where %s = %s ORDERBY %s' % (os.path.join(infile,subtable), tsel[0], tsel[1], ordering))

    # master dataset holders
    mvars, midxs = {}, {}
    cshape, bad_cols = compute_dimensions(sorted_table, ignore)

    row_key, exp_keys = list(keys.keys())[0], list(keys.keys())[1:] if len(keys) > 1 else []
    target_row_key, target_exp_keys = list(keys.values())[0], list(keys.values())[1:] if len(keys) > 1 else []
    rows = np.hstack([sorted_table.getcol(rr)[:,None] for rr in np.atleast_1d(row_key)]).squeeze()
    unique_row_keys, row_changes, row_idxs = np.unique(rows, axis=0, return_index=True, return_inverse=True)
    if unique_row_keys.ndim > 1:  # use index values when grouping columns
        unique_row_keys = np.arange(len(unique_row_keys))
    elif target_row_key in timecols:  # convert to datetime
        unique_row_keys = convert_time(unique_row_keys)
    
    for kk, key in enumerate(exp_keys):
        rows = np.hstack([sorted_table.getcol(rr)[:,None] for rr in np.atleast_1d(key)]).squeeze()
        midxs[target_exp_keys[kk]] = list(np.unique(rows, axis=0, return_inverse=True))
        if midxs[target_exp_keys[kk]][0].ndim > 1:
            midxs[target_exp_keys[kk]][0] = np.arange(len(midxs[target_exp_keys[kk]][0]))
        elif target_exp_keys[kk] in timecols:
            midxs[target_exp_keys[kk]][0] = convert_time(midxs[target_exp_keys[kk]][0])
    
    # store the dimension shapes known so far (grows later on) and the coordinate values
    mdims = dict([(target_row_key, len(unique_row_keys))] + [(mm, midxs[mm][0].shape[0]) for mm in list(midxs.keys())])
    mcoords = dict([(target_row_key, [])] + [(mm, xarray.DataArray(midxs[mm][0], dims=target_exp_keys[ii])) for ii,mm in enumerate(list(midxs.keys()))])
    start = time.time()
    
    # we want to parse the table in batches equal to the specified number of unique row keys, not number of rows
    # so we need to compute the proper number of rows to get the correct number of unique row keys
    for cc, start_idx in enumerate(range(0, len(unique_row_keys), chunks[0])):
        chunk = np.arange(min(chunks[0], len(unique_row_keys) - start_idx)) + start_idx
        end_idx = row_changes[chunk[-1] + 1] if chunk[-1] + 1 < len(row_changes) else len(row_idxs)
        idx_range = np.arange(row_changes[chunk[0]], end_idx)  # indices of table to be read, they are contiguous because table is sorted
        mcoords.update({row_key.lower(): xarray.DataArray(unique_row_keys[chunk], dims=target_row_key)})
        batch = len(unique_row_keys) // chunks[0]
        rtestimate = (' remaining time est %s s'+' '*10) % str(int(((time.time() - start) / cc) * (batch - cc))) if cc > 0 else ''

        for col in sorted_table.colnames():
            if (col in ignore) or (col in bad_cols): continue
            if col in keys.keys(): continue  # skip dim columns (unless they are tuples)
            print('reading chunk %s of %s, col %s...%s' % (str(start_idx // chunks[0]), str(batch), col, rtestimate), end='\r')

            try:
                if col in cshape:   # if this column has a varying shape, it needs to be normalized
                    data = sorted_table.getvarcol(col, idx_range[0], len(idx_range))
                    tshape = cshape[col]  # grab the max dimensionality of this col
                    # expand the variable shaped column to the maximum size of each dimension
                    # blame the person who devised the tb.getvarcol return structure
                    data = np.array([np.pad(data['r' + str(kk)][..., 0], np.stack((tshape * 0, tshape - data['r' + str(kk)].shape[:-1]), -1)).transpose()
                                     for kk in np.arange(start_idx + 1, start_idx + len(data) + 1)])
                else:
                    data = sorted_table.getcol(col, idx_range[0], len(idx_range)).transpose()
            except Exception:
                bad_cols += [col]
                continue
                
            # sometimes, even though the table has >0 rows, certain columns still return 0 rows
            if len(data) == 0: continue
            
            # compute the full shape of this chunk with the expanded dimensions and initialize to NANs
            fullshape = (len(chunk),) + tuple([midxs[mm][0].shape[0] for mm in list(midxs.keys())]) + data.shape[1:]
            fulldata = np.full(fullshape, np.nan, dtype=data.dtype)
            
            # compute the proper location to put the chunk data in the full expanded dimensions and assign
            # some NAN's will likely remain where there are holes in the expanded dimensions
            didxs = (row_idxs[idx_range] - chunk[0],) + tuple([midxs[tt][1][idx_range] for tt in midxs.keys()])
            fulldata[didxs] = data

            # if this column has additional dimensionality beyond the expanded dims, we need to create/reuse placeholder names
            dims = [kk for kk in [target_row_key] + target_exp_keys]
            for ii, dd in enumerate(fullshape[len(keys):]):
                if (ii+len(keys) >= len(mdims)) or (dd not in list(mdims.values())[ii+len(keys):]):
                    mdims['d%i' % len(mdims.keys())] = dd
                dims += [list(mdims.keys())[ii+len(keys):][list(mdims.values())[ii+len(keys):].index(dd)]]
            
            # set chunking based on passed in chunk shape, expanding last dimension if necessary
            chunking = [chunks[di] if di < len(chunks) else chunks[-1] for di, dk in enumerate(dims)]
            chunking = [cs if cs > 0 else fulldata.shape[ci] for ci, cs in enumerate(chunking)]

            # store in list of data variables
            mvars[col.upper()] = xarray.DataArray(fulldata, dims=dims).chunk(dict(zip(dims, chunking)))
        
        
        xds = xarray.Dataset(mvars, coords=mcoords).rename(dimnames)
        if (not nofile) and (start_idx == 0):
            encoding = dict(zip(list(xds.data_vars), cycle([{'compressor': compressor}])))
            if len(bad_cols) > 0: xds = xds.assign_attrs({'bad_cols': bad_cols})
            xds.to_zarr(os.path.join(outfile,subtable), mode='w', encoding=encoding, consolidated=True)
        elif not nofile:
            xds.to_zarr(os.path.join(outfile,subtable), mode='a', append_dim=row_key.lower(), compute=True, consolidated=True)

    sorted_table.close()
    tb_tool.close()
    if not nofile:
        xds = xarray.open_zarr(os.path.join(outfile,subtable))

    return xds
