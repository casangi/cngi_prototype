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
import os
from casacore import tables
from numcodecs import Blosc
import pandas as pd
import xarray
import dask.array
import dask.delayed
import numpy as np
from itertools import cycle
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)



########################################################
# helper for reading time columns to datetime format
# pandas datetimes are referenced against a 0 of 1970-01-01
# CASA's modified julian day reference time is (of course) 1858-11-17
# this requires a correction of 3506716800 seconds which is hardcoded to save time
def convert_time(rawtimes):
    correction = 3506716800.0
    return pd.to_datetime(np.array(rawtimes) - correction, unit='s').values


#######################################################
# helper function extract data chunk for each col
# this is fed to dask.delayed
def read_col_chunk(ts_taql, col, cshape, tidxs, bidxs, didxs, d1, d2):
    ts_tb = tables.taql(ts_taql)
    if (len(cshape) == 2) or (col == 'UVW'):  # all the scalars and UVW
        data = np.array(ts_tb.getcol(col, 0, -1))
    elif len(cshape) == 3:  # WEIGHT, SIGMA
        data = ts_tb.getcolslice(col, d1[0], d1[1], [], 0, -1)
    elif len(cshape) == 4:  # DATA and FLAG
        data = ts_tb.getcolslice(col, (d1[0], d2[0]), (d1[1], d2[1]), [], 0, -1)

    # full data is the maximum of the data shape and chunk shape dimensions
    fulldata = np.full(cshape, np.nan, dtype=data.dtype)
    if len(didxs) > 0: fulldata[tidxs[didxs], bidxs[didxs]] = data[didxs]
    ts_tb.close()
    return fulldata



##################################################################
# read casacore table format in to memory
##################################################################
def read_simple_table(infile, subtable='', timecols=None, ignore=None, add_row_id=False):
    if timecols is None: timecols = []
    if ignore is None: ignore = []

    tb_tool = tables.table(os.path.join(infile, subtable), readonly=True, lockoptions={'option': 'usernoread'}, ack=False)
    if tb_tool.nrows() == 0:
        tb_tool.close()
        return xarray.Dataset()

    dims = ['d%i' % ii for ii in range(20)]
    if (len(subtable) > 0) and add_row_id: dims[0] = subtable.lower() + '_id'

    cols = tb_tool.colnames()
    mvars, mcoords, xds = {}, {}, xarray.Dataset()

    tr = tb_tool.row(ignore, exclude=True)[:]

    # extract data for each col
    for col in cols:
        if (col in ignore) or (not tb_tool.iscelldefined(col,0)): continue
        if tb_tool.coldatatype(col) == 'record': continue   # not supported

        try:
            data = np.stack([rr[col] for rr in tr])
            if isinstance(tr[0][col], dict):
                data = np.stack([rr[col]['array'].reshape(rr[col]['shape']) if len(rr[col]['array']) > 0 else np.array(['']) for rr in tr])
        except:
            # sometimes the columns are variable, so we need to standardize to the largest sizes
            if len(np.unique([isinstance(rr[col], dict) for rr in tr])) > 1: continue   # can't deal with this case
            mshape = np.array(np.max([np.array(rr[col]).shape for rr in tr], axis=0))
            data = np.stack([np.pad(rr[col] if len(rr[col]) > 0 else np.array(rr[col]).reshape(np.arange(len(mshape))*0),
                                    [(0, ss) for ss in mshape - np.array(rr[col]).shape], 'constant', constant_values=np.nan) for rr in tr])

        if len(data) == 0: continue
        if col in timecols: convert_time(data)
        if col.upper().endswith('_ID'):
            mcoords[col.lower()] = xarray.DataArray(data, dims=['d%i_%i' % (di, ds) for di, ds in enumerate(np.array(data).shape)])
        else:
            mvars[col.upper()] = xarray.DataArray(data, dims=['d%i_%i'%(di,ds) for di, ds in enumerate(np.array(data).shape)])

    xds = xarray.Dataset(mvars, coords=mcoords)
    xds = xds.rename(dict([(dv, dims[di]) for di, dv in enumerate(xds.dims)]))
    bad_cols = list(np.setdiff1d([dv.lower() for dv in tb_tool.colnames()], [dv.lower() for dv in list(xds.data_vars)+list(xds.coords)]))
    if len(bad_cols) > 0: xds = xds.assign_attrs({'bad_cols':bad_cols})
    tb_tool.close()

    return xds





#####################################################################
def read_main_table(infile, subsel=0, ignore=None, chunks=(400, 200, 100, 2)):
    if ignore is None: ignore = []

    # select just the specified ddi
    tb_tool = tables.taql('select * from %s where DATA_DESC_ID = %i' % (infile, subsel))
    if tb_tool.nrows() == 0:
        tb_tool.close()
        return xarray.Dataset()

    # main table uses time x (antenna1,antenna2)
    ant1, ant2 = tb_tool.getcol('ANTENNA1',0,-1), tb_tool.getcol('ANTENNA2', 0, -1)
    baselines = np.array([str(ll[0]).zfill(3)+'_'+str(ll[1]).zfill(3) for ll in np.unique(np.hstack([ant1[:,None], ant2[:,None]]), axis=0)])

    dims, cols = ['time','baseline','chan','pol'], tb_tool.colnames()
    tvars, mcoords, xds = {}, {}, xarray.Dataset()

    cshapes = [np.array(tb_tool.getcell(col,0)).shape for col in cols if tb_tool.iscelldefined(col,0)]
    chan_cnt, pol_cnt = [(cc[0],cc[1]) for cc in cshapes if len(cc) == 2][0]

    ts_tb = tables.taql('select DISTINCT TIME from %s where DATA_DESC_ID = %i' % (infile, subsel))
    utimes = np.unique(ts_tb.getcol('TIME',0,-1))
    tol = np.diff(utimes).min()/4  # add a tol around the time ranges returned by taql
    ts_tb.close()

    # loop over time chunks
    for tc in range(0, len(utimes), chunks[0]):
        times = (utimes[tc]-tol, utimes[min(len(utimes)-1, tc+chunks[0]-1)]+tol)
        ctlen = min(len(utimes), tc+chunks[0]) - tc  # chunk time length
        bvars = {}

        # loop over baseline chunks
        for bc in range(0, len(baselines), chunks[1]):
            blines = (baselines[bc], baselines[min(len(baselines)-1, bc+chunks[1]-1)])
            cblen = min(len(baselines) - bc, chunks[1])

            # read the specified chunk of data
            #def read_chunk(infile, subsel, times, blines, chans, pols):
            dtql = 'DATA_DESC_ID = %i' % subsel
            ttql = 'TIME BETWEEN %f and %f' % times
            atql = 'ANTENNA1 BETWEEN %i and %i' % (int(blines[0].split('_')[0]), int(blines[1].split('_')[0]))
            ts_taql = 'select * from %s where %s AND %s AND %s' % (infile, dtql, ttql, atql)
            ts_tb = tables.taql(ts_taql)

            tidxs = np.searchsorted(utimes, ts_tb.getcol('TIME', 0, -1)) - tc
            ts_ant1, ts_ant2 = ts_tb.getcol('ANTENNA1', 0, -1), ts_tb.getcol('ANTENNA2', 0, -1)
            ts_bases = [str(ll[0]).zfill(3) + '_' + str(ll[1]).zfill(3) for ll in np.hstack([ts_ant1[:, None], ts_ant2[:, None]])]
            bidxs = np.searchsorted(baselines, ts_bases) - bc

            # some antenna 2's will be out of bounds for this chunk, store rows that are in bounds
            didxs = np.where((bidxs >= 0) & (bidxs < min(chunks[1], len(baselines)-bc)))[0]

            # loop over each column and create delayed dask arrays
            for col in cols:
                if (col in ignore + ['TIME']) or (not tb_tool.iscelldefined(col, 0)): continue
                if col not in bvars: bvars[col] = []

                cdata = tb_tool.getcol(col, 0, 1)[0]
                if len(cdata.shape) == 0:
                    delayed_array = dask.delayed(read_col_chunk)(ts_taql, col, (ctlen, cblen), tidxs, bidxs, didxs, None, None)
                    bvars[col] += [dask.array.from_delayed(delayed_array, (ctlen, cblen), cdata.dtype)]

                elif col == 'UVW':
                    delayed_array = dask.delayed(read_col_chunk)(ts_taql, col, (ctlen, cblen, 3), tidxs, bidxs, didxs, None, None)
                    bvars[col] += [dask.array.from_delayed(delayed_array, (ctlen, cblen, 3), cdata.dtype)]

                elif len(cdata.shape) == 1:
                    pol_list = []
                    dd = 2 if cdata.shape == chan_cnt else 3
                    for pc in range(0, cdata.shape[0], chunks[dd]):
                        pols = (pc, min(cdata.shape[0], pc + chunks[dd]) - 1)
                        cshape = (ctlen, cblen,) + (pols[1]-pols[0]+1,)
                        delayed_array = dask.delayed(read_col_chunk)(ts_taql, col, cshape, tidxs, bidxs, didxs, pols, None)
                        pol_list += [dask.array.from_delayed(delayed_array, cshape, cdata.dtype)]
                    bvars[col] += [dask.array.concatenate(pol_list, axis=2)]

                elif len(cdata.shape) == 2:
                    chan_list = []
                    for cc in range(0, cdata.shape[0], chunks[2]):
                        chans = (cc, min(cdata.shape[0],cc+chunks[2])-1)
                        pol_list = []
                        for pc in range(0, cdata.shape[1], chunks[3]):
                            pols = (pc, min(cdata.shape[1], pc+chunks[3])-1)
                            cshape = (ctlen, cblen,) + (chans[1]-chans[0]+1, pols[1]-pols[0]+1)
                            delayed_array= dask.delayed(read_col_chunk)(ts_taql, col, cshape, tidxs, bidxs, didxs, chans, pols)
                            pol_list += [dask.array.from_delayed(delayed_array, cshape, cdata.dtype)]
                        chan_list += [dask.array.concatenate(pol_list, axis=3)]
                    bvars[col] += [dask.array.concatenate(chan_list, axis=2)]
            ts_tb.close()

        # now concat all the dask chunks from each baseline
        for kk in bvars.keys():
            if len(bvars[kk]) == 0: continue
            if kk not in tvars: tvars[kk] = []
            tvars[kk] += [dask.array.concatenate(bvars[kk], axis=1)]

    # now concat all the dask chunks from each time to make the xds
    mvars = {}
    for kk in tvars.keys():
        if kk == 'UVW':
            mvars[kk.upper()] = xarray.DataArray(dask.array.concatenate(tvars[kk], axis=0), dims=dims[:2] + ['uvw_index'])
        elif len(tvars[kk][0].shape) == 3 and (tvars[kk][0].shape[-1] == pol_cnt):
            mvars[kk.upper()] = xarray.DataArray(dask.array.concatenate(tvars[kk], axis=0), dims=dims[:2] + ['pol'])
        elif len(tvars[kk][0].shape) == 3 and (tvars[kk][0].shape[-1] == chan_cnt):
            mvars[kk.upper()] = xarray.DataArray(dask.array.concatenate(tvars[kk], axis=0), dims=dims[:2] + ['chan'])
        else:
            mvars[kk.upper()] = xarray.DataArray(dask.array.concatenate(tvars[kk], axis=0), dims=dims[:len(tvars[kk][0].shape)])

    mcoords['time'] = xarray.DataArray(convert_time(utimes), dims=['time'])
    mcoords['baseline'] = xarray.DataArray(np.arange(len(baselines)), dims=['baseline'])
    xds = xarray.Dataset(mvars, coords=mcoords)
    tb_tool.close()

    return xds






#####################################################################
def read_pointing_table(infile, chunks=(100, 100, 20, 20)):
    tb_tool = tables.taql('select * from %s' % infile)
    if tb_tool.nrows() == 0:
        tb_tool.close()
        return xarray.Dataset()

    # pointing table uses time x antenna_id
    baselines = np.unique(tb_tool.getcol('ANTENNA_ID',0,-1))

    dims, cols = ['time','antenna_id','d2','d3'], tb_tool.colnames()
    tvars, mcoords, xds = {}, {}, xarray.Dataset()

    ts_tb = tables.taql('select DISTINCT TIME from %s' % infile)
    utimes = np.unique(ts_tb.getcol('TIME',0,-1))
    tol = np.diff(utimes).min()/4  # add a tol around the time ranges returned by taql
    ts_tb.close()

    # loop over time chunks
    for tc in range(0, len(utimes), chunks[0]):
        times = (utimes[tc]-tol, utimes[min(len(utimes)-1, tc+chunks[0]-1)]+tol)
        ctlen = min(len(utimes), tc+chunks[0]) - tc  # chunk time length
        bvars = {}

        # loop over antenna_id chunks
        for bc in range(0, len(baselines), chunks[1]):
            blines = (baselines[bc], baselines[min(len(baselines)-1, bc+chunks[1]-1)])
            cblen = min(len(baselines) - bc, chunks[1])

            # read the specified chunk of data
            ttql = 'TIME BETWEEN %f and %f' % times
            atql = 'ANTENNA_ID BETWEEN %i and %i' % blines
            ts_taql = 'select * from %s where %s AND %s' % (infile, ttql, atql)
            ts_tb = tables.taql(ts_taql)

            tidxs = np.searchsorted(utimes, ts_tb.getcol('TIME', 0, -1)) - tc
            bidxs = np.searchsorted(baselines, ts_tb.getcol('ANTENNA_ID', 0, -1)) - bc
            didxs = np.arange(len(bidxs))

            # loop over each column and create delayed dask arrays
            for col in cols:
                if (col in ['TIME', 'ANTENNA_ID']) or (not tb_tool.iscelldefined(col, 0)): continue
                if col not in bvars: bvars[col] = []

                cdata = tb_tool.getcol(col, 0, 1)[0]
                if isinstance(cdata, str): cdata = np.array(cdata)
                if len(cdata.shape) == 0:
                    delayed_array = dask.delayed(read_col_chunk)(ts_taql, col, (ctlen, cblen), tidxs, bidxs, didxs, None, None)
                    bvars[col] += [dask.array.from_delayed(delayed_array, (ctlen, cblen), cdata.dtype)]

                elif len(cdata.shape) == 2:
                    d1_list = []
                    for cc in range(0, cdata.shape[0], chunks[2]):
                        d1s = (cc, min(cdata.shape[0],cc+chunks[2])-1)
                        d2_list = []
                        for pc in range(0, cdata.shape[1], chunks[3]):
                            d2s = (pc, min(cdata.shape[1], pc+chunks[3])-1)
                            cshape = (ctlen, cblen,) + (d1s[1]-d1s[0]+1, d2s[1]-d2s[0]+1)
                            delayed_array= dask.delayed(read_col_chunk)(ts_taql, col, cshape, tidxs, bidxs, didxs, d1s, d2s)
                            d2_list += [dask.array.from_delayed(delayed_array, cshape, cdata.dtype)]
                        d1_list += [dask.array.concatenate(d2_list, axis=3)]
                    bvars[col] += [dask.array.concatenate(d1_list, axis=2)]
            ts_tb.close()

        # now concat all the dask chunks from each baseline
        for kk in bvars.keys():
            if len(bvars[kk]) == 0: continue
            if kk not in tvars: tvars[kk] = []
            tvars[kk] += [dask.array.concatenate(bvars[kk], axis=1)]

    # now concat all the dask chunks from each time to make the xds
    mvars = {}
    for kk in tvars.keys():
        mvars[kk.upper()] = xarray.DataArray(dask.array.concatenate(tvars[kk], axis=0), dims=dims[:len(tvars[kk][0].shape)])

    mcoords['time'] = xarray.DataArray(convert_time(utimes), dims=['time'])
    mcoords['antenna_id'] = xarray.DataArray(np.arange(len(baselines)), dims=['antenna_id'])
    xds = xarray.Dataset(mvars, coords=mcoords)
    tb_tool.close()

    return xds









##################################################################
# convert a legacy casacore table format to CNGI xarray/zarr
##################################################################
def convert_simple_table(infile, outfile, subtable='', timecols=None, ignore=None, compressor=None, chunks=(2000, -1), nofile=False, add_row_id=False):
    if timecols is None: timecols = []
    if ignore is None: ignore = []
    if compressor is None: compressor = Blosc(cname='zstd', clevel=2, shuffle=0)

    tb_tool = tables.table(os.path.join(infile, subtable), readonly=True, lockoptions={'option': 'usernoread'}, ack=False)

    # handle no chunking of first axis and ensure enough dims are covered
    dims = ['d%i'%ii for ii in range(20)]
    chunks = chunks + tuple(np.repeat(chunks[-1], 10))
    if (len(subtable) > 0) and add_row_id: dims[0] = subtable.lower() + '_id'
    if chunks[0] == -1: chunks = (tb_tool.nrows(),) + chunks[1:]

    # master dataset holders
    cols = tb_tool.colnames()
    mvars, mcoords, xds = {}, {}, xarray.Dataset()

    for ii, rr in enumerate(range(0, tb_tool.nrows(), chunks[0])):
        tr = tb_tool.row(ignore, exclude=True)[rr:rr+chunks[0]]

        # extract data for each col
        for col in cols:
            if (col in ignore) or (not tb_tool.iscelldefined(col,0)): continue
            try:
                data = np.stack([rr[col] for rr in tr])
                if isinstance(tr[0][col], dict):
                    data = np.stack([rr[col]['array'].reshape(rr[col]['shape']) if len(rr[col]['array']) > 0 else np.array(['']) for rr in tr])
            except:
                # sometimes the columns are variable, so we need to standardize to the largest sizes
                mshape = np.max([np.array(rr[col]).shape for rr in tr], axis=0)
                data = np.stack([np.pad(rr[col], [(0,ss) for ss in mshape - np.array(rr[col]).shape], 'constant', constant_values=np.nan) for rr in tr])

            if data.size == 0: continue
            if col in timecols: convert_time(data)
            chunking = tuple([chunks[ii] if chunks[ii] > 0 else data.shape[ii] for ii in range(len(data.shape))])
            if col.upper().endswith('_ID'):
                mcoords[col.lower()] = xarray.DataArray(data, dims=['d%i_%i' % (di, ds) for di, ds in enumerate(data.shape)])
            else:
                mvars[col.upper()] = xarray.DataArray(data, dims=['d%i_%i'%(di,ds) for di, ds in enumerate(data.shape)]).chunk(chunking)

        xds = xarray.Dataset(mvars, coords=mcoords)
        xds = xds.rename(dict([(dv, dims[di]) for di, dv in enumerate(xds.dims)]))
        if (not nofile) and (rr == 0):
            encoding = dict(zip(list(xds.data_vars), cycle([{'compressor': compressor}])))
            xds.to_zarr(os.path.join(outfile, subtable), mode='w', encoding=encoding, consolidated=True)
        elif not nofile:
            xds.to_zarr(os.path.join(outfile, subtable), mode='a', append_dim='time', compute=True, consolidated=True)
        #print('processed %s chunks' % str(ii+1), end='\r')

    if (not nofile) and (tb_tool.nrows() > 0):
        xds = xarray.open_zarr(os.path.join(outfile, subtable))
    tb_tool.close()

    return xds






#############################################################################
# convert a legacy casacore main table or POINTING table to CNGI xarray/zarr
# with dimension expansion from splitting apart rows by values in specified columns
#
# subsel = DDI to convert (int)
#############################################################################
def convert_expanded_table(infile, outfile, subsel=None, ignore=None, compressor=None, chunks=(100, 100, 20, 2), nofile=False):
    if ignore is None: ignore = []
    if compressor is None: compressor = Blosc(cname='zstd', clevel=2, shuffle=0)

    # select just the specified ddi
    subseltql = '' if subsel is None else ' where DATA_DESC_ID = %s' % str(subsel)
    tb_tool = tables.taql('select * from %s%s' % (infile, subseltql))

    # handle no chunking of first axis
    chunks = [chunks[di] if di < len(chunks) else chunks[-1] for di in range(4)]
    if nofile or (chunks[0] == -1):
        chunks[0] = tb_tool.nrows()

    # pointing table will use dims of time by antenna_id
    # main table uses time x (antenna1,antenna2)
    if infile.endswith('POINTING'):
        baselines = np.unique(tb_tool.getcol('ANTENNA_ID',0,-1))
        mcoords = dict([('time', []), ('antenna_id', xarray.DataArray(baselines, dims=['antenna_id']))])
    else:
        ant1, ant2 = tb_tool.getcol('ANTENNA1',0,-1), tb_tool.getcol('ANTENNA2', 0, -1)
        baselines = np.array([str(ll[0]).zfill(3)+'_'+str(ll[1]).zfill(3) for ll in np.unique(np.hstack([ant1[:,None], ant2[:,None]]), axis=0)])
        mcoords = dict([('time', []), ('baseline', xarray.DataArray(np.arange(len(baselines)).astype('int64'), dims=['baseline']))])

    dims = ['time', 'antenna_id', 'd2', 'd3'] if infile.endswith('POINTING') else ['time','baseline','chan','pol']
    cols = tb_tool.colnames()
    times, mvars, mchunks, xds = [], {}, {}, xarray.Dataset()

    ts_tb = tables.taql('select DISTINCT TIME from %s%s' % (infile, subseltql))
    utimes = np.unique(ts_tb.getcol('TIME',0,-1))
    ts_tb.close()
    subseltql = ' where ' if subsel is None else ' where DATA_DESC_ID = %s AND ' % str(subsel)

    # for each batch of times in a chunk
    for ts in range(0, len(utimes), chunks[0]):
        ts_tb = tables.taql('select * from %s%sTIME BETWEEN %f and %f' % (infile, subseltql, utimes[ts], utimes[min(len(utimes)-1,ts+chunks[0]-1)]))
        raw_times = ts_tb.getcol('TIME',0,-1)
        times = np.unique(raw_times)
        tidxs = np.searchsorted(utimes, raw_times)-ts

        # compute the indices of all the times and baselines at this timestep
        if infile.endswith('POINTING'):
            ts_bases = ts_tb.getcol('ANTENNA_ID',0,-1)
        else:
            ts_ant1, ts_ant2 = ts_tb.getcol('ANTENNA1',0,-1), ts_tb.getcol('ANTENNA2',0,-1)
            ts_bases = np.array([str(ll[0]).zfill(3) + '_' + str(ll[1]).zfill(3) for ll in np.hstack([ts_ant1[:,None], ts_ant2[:,None]])])
        bidxs = np.searchsorted(baselines, ts_bases)

        # extract data for each col
        for col in cols:
            if (col in ignore + ['TIME']) or (not ts_tb.iscelldefined(col,0)): continue
            data = np.array(ts_tb.getcol(col,0,-1))
            if data.size == 0: continue
            fulldata = np.full((len(times), len(baselines),) + data.shape[1:], np.nan, dtype=data.dtype)
            fulldata[tidxs,bidxs] = data
            mchunks = dict(zip(dims[:len(fulldata.shape)], [chunks[ii] if chunks[ii] > 0 else fulldata.shape[ii] for ii in range(len(fulldata.shape))]))
            mvars[col.upper()] = xarray.DataArray(fulldata, dims=dims[:len(fulldata.shape)]).chunk(mchunks)

        mcoords.update({'time': xarray.DataArray(convert_time(np.array(times)), dims=['time'])})
        xds = xarray.Dataset(mvars, coords=mcoords)
        if (not nofile) and (ts == 0):
            encoding = dict(zip(list(xds.data_vars), cycle([{'compressor': compressor}])))
            xds.to_zarr(outfile, mode='w', encoding=encoding, consolidated=True)
        elif not nofile:
            xds.to_zarr(outfile, mode='a', append_dim='time', compute=True, consolidated=True)
        print('processed %s time steps' % str(ts+1), end='\r')
        ts_tb.close()

    if (not nofile) and (tb_tool.nrows() > 0):
        xds = xarray.open_zarr(outfile)
    tb_tool.close()

    return xds

