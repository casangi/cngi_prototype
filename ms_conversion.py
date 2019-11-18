from cngi.direct import InitializeFramework
from cngi.conversion import ms_to_pq


if __name__ == '__main__':
  #client = InitializeFramework(workers=4, memory='8GB')
  
  ms_to_pq('/export/data_1/data/uid___A002_Xc3032e_X27c3.ms', ddi=25)
