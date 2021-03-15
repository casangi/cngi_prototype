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


def selectedchannels(chans=None, shapeLength=64):
    """
        This method returns all the selected zero-based channel numbers from the specified string within the image.

        Parameters
        ----------
        chans : the specified string defined chans
            input string
        length :
            input int

        Returns
        -------
        This method returns all the selected zero-based channel numbers from the specified string within the image.

        Examples
        chans = "3~8, 54~60"

        """
    import re
    import numpy as np

    x = []
    #split string into substrings
    if(chans.find(',') != -1):
        n1 = re.split(r',', chans)
    elif(chans.find(';') != -1):
        n1 = re.split(r';', chans)
    else:
        n1=[chans]

    for s in n1:
        n2 = re.findall("\d+", s)
        if ( s.find('~') != -1):
            x += [i for i in range(max(0,int(n2[0])), min(int(n2[1])+1, shapeLength))]
        elif (s.find('>') != -1):
            x += [i for i in range(max(0,int(n2[0])+1), shapeLength)]
        elif (s.find('<') != -1):
            x += [i for i in range(0, min(int(n2[0]),shapeLength))]
        else:
            x += [int(n2[0])]

    return x
