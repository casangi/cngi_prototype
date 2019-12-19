from casatools import image as ia
from cngi.dio import read_image

IA = ia()
rc = IA.open('~/dev/data/ALMA_smallcube.image.fits') 
xds = read_image('~/dev/data/ALMA_smallcube.image.zarr')

points = [(1,1),(24,112),(11,500),(340,223),(503,101),(511,511)]

# position
for pt in points:
  casa_coords = IA.toworld(np.array(pt))['numeric'][:2]
  cngi_coords = [xds.right_ascension.values[pt], xds.declination.values[pt]]
  percent_dev = (casa_coords - cngi_coords)/casa_coords * 100
  print('ra/dec deviation % : ', percent_dev)

# stokes
for pt in points:
  casa_coords = IA.toworld(np.array(pt))['numeric'][2]
  cngi_coords = xds.image[pt].stokes.values[0]
  percent_dev = (casa_coords - cngi_coords)/casa_coords * 100
  print('stokes deviation % : ', percent_dev)

# frequency
for pt in points:
  casa_coords = []
  for ch in range(xds.frequency.shape[0]):
    casa_coords += [IA.toworld(np.array(pt+(0,ch)))['numeric'][3]]
  cngi_coords = xds.image[pt].frequency.values
  percent_dev = (np.array(casa_coords) - cngi_coords)/np.array(casa_coords) * 100
  print('frequency deviation % : ', percent_dev)
