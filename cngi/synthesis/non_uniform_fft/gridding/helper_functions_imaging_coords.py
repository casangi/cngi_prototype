import numpy as np

def _coordinates(npixel: int):
    """ 1D array which spans [-.5,.5[ with 0 at position npixel/2
    """
    return (np.arange(npixel) - npixel // 2) / npixel


def _coordinates2(npixel: int):
    """Two dimensional grids of coordinates spanning -1 to 1 in each dimension
    1. a step size of 2/npixel and
    2. (0,0) at pixel (floor(n/2),floor(n/2))
    """
    return (np.mgrid[0:npixel, 0:npixel] - npixel // 2) / npixel