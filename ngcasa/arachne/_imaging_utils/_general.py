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
        
