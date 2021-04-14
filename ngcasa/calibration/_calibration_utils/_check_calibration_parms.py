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

#ducting - code is complex and might fail after some time if parameters is wrong. Sensable values are also checked. Gives printout of all wrong parameters. Dirty images alone has 14 parameters.

import numpy as np
from  cngi._utils._check_parms import _check_parms, _check_dataset

def _check_self_cal(solve_parms):
    import numbers
    parms_passed = True
    if not(_check_parms(solve_parms, 'gaintype', [str], acceptable_data=['G','T'], default='T')): parms_passed = False
    if not(_check_parms(solve_parms, 'solint', [str], acceptable_data=['int','all'], default='int')): parms_passed = False
    if not(_check_parms(solve_parms, 'refant_id', [int], default=-1)): parms_passed = False
    if not(_check_parms(solve_parms, 'phase_only', [bool], default=False)): parms_passed = False
    if not(_check_parms(solve_parms, 'minsnr', [numbers.Number], default=0.0)): parms_passed = False
    if not(_check_parms(solve_parms, 'minblperant', [int], default=4)): parms_passed = False
    if not(_check_parms(solve_parms, 'ginfo', [bool], default=False)): parms_passed = False
    
    
    if solve_parms['refant_id'] == -1:
        solve_parms['refant_id'] = None
    return parms_passed
