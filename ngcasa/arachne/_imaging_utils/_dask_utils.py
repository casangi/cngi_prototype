import pandas as pd
import dask
import dask.array as da

def _find_unique_subset(a,b):
    a_pd = pd.DataFrame(a)
    b_pd = pd.DataFrame(b)
    
    a_pd = a_pd.append(b_pd)
    
    a_pd = a_pd.drop_duplicates(a_pd.columns[-1])
    #print(a_pd.columns[-1])
    return a_pd.to_numpy()
    #return da.from_array(a_pd.to_numpy(),chunks=chunks)

def _tree_combine_list(list_to_sum,func):
    import dask.array as da
    while len(list_to_sum) > 1:
        new_list_to_sum = []
        for i in range(0, len(list_to_sum), 2):
            if i < len(list_to_sum) - 1:
                lazy = dask.delayed(_find_unique_subset)(list_to_sum[i],list_to_sum[i+1])
            else:
                lazy = list_to_sum[i]
            new_list_to_sum.append(lazy)
        list_to_sum = new_list_to_sum
    return list_to_sum[0]
