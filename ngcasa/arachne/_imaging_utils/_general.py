
#Rememeber to get a list with single dim use shape=(x,)
def _ndim_list(shape):
    return [_ndim_list(shape[1:]) if len(shape) > 1 else None for _ in range(shape[0])]
  
#
#def _ndim_list(shape):
#    print('shape',shape)
#    print('shape len',len(shape))
#    if len(shape) > 1:
#        print('1')
#        return [_ndim_list(shape[1:])]
#    else:
#        print('2')
#        return [None for _ in range(shape[0])]
        
