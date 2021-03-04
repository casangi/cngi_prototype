# requires a data/refimager directory
# obtain from https://drive.google.com/file/d/1LfEtb5POmAS3Qv-CZ1_01RNtPCltTAMD/view?usp=sharing

import os
from cngi.conversion import convert_ms

ms_list = [ms for ms in os.listdir('data/refimager') if ms.endswith('.ms')]

for ms in ms_list:
    print('##### Converting %s #####' % ms.split('/')[-1])
    mxds = convert_ms('data/refimager/'+ms)
    

#mxds = convert_ms('data/sis14_twhya_calibrated_flagged.ms')
#mxds = convert_ms('data/M100_full.ms')

mxds = convert_ms('data/uid___A002_Xc3032e_X27c3.ms', ddis=[0,1,2,3,4,5,6,7,8,9,10,11,12,'global'])
