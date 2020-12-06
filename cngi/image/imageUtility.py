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
