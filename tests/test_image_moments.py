import os, warnings
warnings.simplefilter("ignore", category=RuntimeWarning)  # suppress warnings about nan-slices

from casatasks import tclean
from casatasks import immoments
from cngi.conversion import convert_image
from cngi.image import implot
from cngi.image import moments
import os
import numpy as np

#tclean returns error for some reason 2020 1013
'''
tclean(vis='sis14_twhya_calibrated_flagged.ms', imagename='sis14_twhya_calibrated_flagged', field='5', spw='',
       specmode='cube', deconvolver='hogbom', nterms=1, imsize=[250,250], gridder='standard', cell=['0.1arcsec'],
       nchan=10, weighting='natural', threshold='0mJy', niter=5000, interactive=False, savemodel='modelcolumn',
       usemask='auto-multithresh')
'''
#load image instead of running tclean. Change the file path when test it in your local machine
Casa6ImageFileName = '/Users/wxiong/NRAO/dev/cngi_prototype/data/sis14_twhya_calibrated_flagged/sis14_twhya_calibrated_flagged.image'

# casa6
os.system('rm -rf casa6.immoments.image*')
immoments(Casa6ImageFileName, axis  = 'spectral', moments = [-1,0,1,2,3,5,6,7,8,9,10,11], outfile='casa6.immoments.image')
casa_xds = convert_image('casa6.immoments.image.average', artifacts=['average', 'median','integrated', 'maximum', 'maximum_coord', 'minimum', 'minimum_coord', 'rms', 'standard_deviation', 'weighted_coord', 'weighted_dispersion_coord','abs_mean_dev'])
print(casa_xds)

#NGCI
image_xds=convert_image(Casa6ImageFileName, artifacts=['image'])
cngi_xds = moments(image_xds, moments = [-1,0,1,2,3,5,6,7,8,9,10,11])
print("cngi_xds")

#Delta

#Delta of Moments_Average between casa6 and ngcasa
implot(cngi_xds.MOMENTS_AVERAGE.where(casa_xds.MASK)- casa_xds.IMAGE.where(casa_xds.MASK))

#Delta of Moments_Integrated value between casa6 and ngcasa
implot(cngi_xds.MOMENTS_INTERGRATED.where(casa_xds.MASK)- casa_xds.INTEGRATED.where(casa_xds.MASK))

#Delta of Moments_intensity_weighted_coordinate between casa6 and ngcasa
implot(cngi_xds.MOMENTS_WEIGHTED_COORD.where(casa_xds.MASK))
implot(casa_xds.WEIGHTED_COORD.where(casa_xds.MASK))
implot(cngi_xds.MOMENTS_WEIGHTED_COORD.where(casa_xds.MASK)-casa_xds.WEIGHTED_COORD.where(casa_xds.MASK))

#Delta of Moments_intensity_weighted_coordinate between casa6 and ngcasa
implot(cngi_xds.MOMENTS_WEIGHTED_DISPERSION_COORD.where(casa_xds.WEIGHTED_DISPERSION_COORD))
implot(casa_xds.WEIGHTED_DISPERSION_COORD.where(casa_xds.MASK))
implot(cngi_xds.MOMENTS_WEIGHTED_DISPERSION_COORD.where(casa_xds.WEIGHTED_DISPERSION_COORD)- casa_xds.WEIGHTED_DISPERSION_COORD.where(casa_xds.MASK))

#Delta of Moments_Median between casa6 and ngcasa
implot(cngi_xds.MOMENTS_MEDIAN.where(casa_xds.MASK)- casa_xds.MEDIAN.where(casa_xds.MASK))

#Delta of Moments_Median_Coordinate between casa6 and ngcasa
#implot(cngi_xds.MOMENTS_MEDIAN_COORD.where(casa_xds.MASK)- casa_xds.IMAGE.where(casa_xds.MASK))

#Delta of Moments_standard_deviation between casa6 and ngcasa
implot(cngi_xds.MOMENTS_STANDARD_DEVIATION.where(casa_xds.MASK))
implot( casa_xds.STANDARD_DEVIATION.where(casa_xds.MASK))
implot(cngi_xds.MOMENTS_STANDARD_DEVIATION.where(casa_xds.MASK)- casa_xds.STANDARD_DEVIATION.where(casa_xds.MASK))

#Delta of Moments_root mean square between casa6 and ngcasa
implot(cngi_xds.MOMENTS_RMS.where(casa_xds.MASK)- casa_xds.RMS.where(casa_xds.MASK))

#Delta of Moments_absolute_mean_deviation between casa6 and ngcasa
implot(cngi_xds.MOMENTS_ABS_MEAN_DEV.where(casa_xds.MASK)- casa_xds.ABS_MEAN_DEV.where(casa_xds.MASK))

#Delta of Moments_Maximum between casa6 and ngcasa
implot(cngi_xds.MOMENTS_MAXIMUM.where(casa_xds.MASK))
implot( casa_xds.MAXIMUM.where(casa_xds.MASK))
implot(cngi_xds.MOMENTS_MAXIMUM.where(casa_xds.MASK)- casa_xds.MAXIMUM.where(casa_xds.MASK))

#Delta of Moments_Maximum_coordinate between casa6 and ngcasa
implot(cngi_xds.MOMENTS_MAXIMUM_COORD.where(casa_xds.MASK))
implot(casa_xds.MAXIMUM_COORD.where(casa_xds.MASK))
implot(cngi_xds.MOMENTS_MAXIMUM_COORD.where(casa_xds.MASK)- casa_xds.MAXIMUM_COORD.where(casa_xds.MASK))

#Delta of Moments_Minimum between casa6 and ngcasa
implot(cngi_xds.MOMENTS_MINIMUM.where(casa_xds.MASK))
implot(casa_xds.MINIMUM.where(casa_xds.MASK))
implot(cngi_xds.MOMENTS_MINIMUM.where(casa_xds.MASK)- casa_xds.MINIMUM.where(casa_xds.MASK))

#Delta of Moments_Average_coordinate between casa6 and ngcasa
implot(cngi_xds.MOMENTS_MINIMUM_COORD.where(casa_xds.MASK))
implot(casa_xds.MINIMUM_COORD.where(casa_xds.MASK))
implot(cngi_xds.MOMENTS_MINIMUM_COORD.where(casa_xds.MASK)- casa_xds.MINIMUM_COORD.where(casa_xds.MASK))


print('finished')