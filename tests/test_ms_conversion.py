# requires a data/refimager directory
# obtain from https://drive.google.com/file/d/1LfEtb5POmAS3Qv-CZ1_01RNtPCltTAMD/view?usp=sharing

import os
import time
import numpy as np
from cngi.conversion import convert_ms, read_ms, describe_ms
from cngi.dio import write_vis


def compare_xds(xds1, xds2):
    match = True
    if not xds1.equals(xds2):
        for col in xds1.data_vars:
            if xds1[col].equals(xds2[col]): # and xds1[col].equals(xds3[col]):
                #print('- MATCH %s' % col)
                continue
            if (xds1[col].dtype == np.int32) or (xds1[col].dtype == np.int64):
                diff = np.max(np.abs(xds1[col].clip(0) - xds2[col].clip(0)))
                #diff = max(diff, np.max(np.abs(xds2[col].values.clip(0) - xds3[col].values.clip(0))))
            elif xds1[col].dtype == np.bool:
                diff = np.max(np.abs(xds1[col] != xds2[col]))
                #diff = float(max(diff, np.max(np.abs(xds2[col].values != xds3[col].values))))
            else:
                diff = np.max(np.abs(np.nan_to_num(xds1[col]) - np.nan_to_num(xds2[col])))
                #diff = max(diff, np.max(np.abs(np.nan_to_num(xds2[col].values) - np.nan_to_num(xds3[col].values))))
            if diff > 1e-8:
                print('+++++ difference found in col %s = %f +++++' % (col, diff))
                match = False
            else:
                #print('- MANUAL MATCH  %s' % col)
                pass
    if match: print('+++++ MATCH +++++')



ms_list = [ms for ms in os.listdir('data/refimager') if ms.endswith('.ms')]

for ms in ms_list:
    print('Converting %s...' % ms.split('/')[-1])
    xds1 = convert_ms('data/refimager/'+ms, ddis=[0]).xds0
    xds2 = read_ms('data/refimager/' + ms, ddis=[0]).xds0
    compare_xds(xds1, xds2)
    mxds2 = read_ms('data/refimager/'+ms)



xds1 = convert_ms('data/sis14_twhya_calibrated_flagged.ms', ddis=[0]).xds0
xds2 = read_ms('data/sis14_twhya_calibrated_flagged.ms', ddis=[0]).xds0
compare_xds(xds1, xds2)

mxds = read_ms('data/sis14_twhya_calibrated_flagged.ms')
write_vis(mxds, 'data/test_conversion.vis.zarr')

os.system('rm -fr data/test_conversion.vis.zarr')

xds1 = convert_ms('data/M100.ms', ddis=[0], chunks=(100,60,16,2)).xds0
xds2 = read_ms('data/M100.ms', ddis=[0], chunks=(100,60,16,2)).xds0
compare_xds(xds1, xds2)

describe_ms('data/M100.ms')

mxds = read_ms('data/M100.ms', ddis=[0,1,2,3], chunks=(100,60,16,2))
write_vis(mxds, 'data/test_conversion.vis.zarr')


xds1 = convert_ms('data/IRAS16293_Band9.ms', ddis=[0], chunks=(100, 100, 300, 2)).xds0
xds2 = read_ms('data/IRAS16293_Band9.ms', ddis=[0], chunks=(100, 100, 300, 2)).xds0
compare_xds(xds1, xds2)

mxds = read_ms('data/IRAS16293_Band9.ms')
write_vis(mxds, 'data/test_conversion.vis.zarr')

xds1 = convert_ms('data/VLASS_J1448_1620.ms', ddis=[2]).xds2
xds2 = read_ms('data/VLASS_J1448_1620.ms', ddis=[2]).xds2
compare_xds(xds1, xds2)

mxds = read_ms('data/VLASS_J1448_1620.ms')
write_vis(mxds, 'data/test_conversion.vis.zarr')


xds1 = convert_ms('data/3c391_ctm_mosaic_10s_spw0.ms', ddis=[0]).xds0
xds2 = read_ms('data/3c391_ctm_mosaic_10s_spw0.ms', ddis=[0]).xds0
compare_xds(xds1, xds2)

mxds = read_ms('data/3c391_ctm_mosaic_10s_spw0.ms')
write_vis(mxds, 'data/test_conversion.vis.zarr')


xds1 = convert_ms('data/uid___A002_Xc3032e_X27c3.ms', ddis=[0], chunks=(1000,400,-1,-1)).xds0
xds2 = read_ms('data/uid___A002_Xc3032e_X27c3.ms', ddis=[0], chunks=(1000,400,1,2)).xds0
compare_xds(xds1, xds2)



describe_ms('data/uid___A002_Xc3032e_X27c3.ms')

start = time.time()
mxds = read_ms('data/uid___A002_Xc3032e_X27c3.ms', chunks=(600,100,400,2))
#xds = read_ms('data/uid___A002_Xc3032e_X27c3.ms', ddis=[25], chunks=(600,100,400,2)).xds25
print('read complete in %s seconds' % str(time.time()-start))
write_vis(mxds, 'data/test_conversion.vis.zarr')
