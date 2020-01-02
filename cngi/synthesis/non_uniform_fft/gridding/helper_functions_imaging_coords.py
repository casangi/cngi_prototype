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