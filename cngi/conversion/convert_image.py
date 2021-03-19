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
"""
this module will be included in the api
"""


##########################################
def convert_image(infile, outfile=None, artifacts=[], compressor=None, chunks=(-1, -1, 1, 1)):
    """
    Convert legacy CASA or FITS format Image to xarray Image Dataset and zarr storage format

    This function requires CASA6 casatools module.

    Parameters
    ----------
    infile : str
        Input image filename (.image or .fits format). If taylor terms are present, they should be in the form of filename.image.tt0 and
        this infile string should be filename.image
    outfile : str
        Output zarr filename. If None, will use infile name with .img.zarr extension
    artifacts : list of str
        List of other image artifacts to include if present with infile. Use None for just the specified infile.
        Default [] uses ``['mask','model','pb','psf','residual','sumwt','weight']``
    compressor : numcodecs.blosc.Blosc
        The blosc compressor to use when saving the converted data to disk using zarr.
        If None the zstd compression algorithm used with compression level 2.
    chunks: 4-D tuple of ints
        Shape of desired chunking in the form of (l, m, channels, polarization), use -1 for entire axis in one chunk. Default is (-1, -1, 1, 1)
        Note: chunk size is the product of the four numbers (up to the actual size of the dimension)

    Returns
    -------
    xarray.core.dataset.Dataset
        new xarray Datasets of Image data contents
    """
    from casatools import image as ia
    from casatools import quanta as qa
    from cngi._utils._table_conversion import convert_simple_table, convert_time
    import numpy as np
    from itertools import cycle
    import importlib_metadata
    import xarray
    from numcodecs import Blosc
    import time, os, warnings, re
    warnings.simplefilter("ignore", category=FutureWarning)  # suppress noisy warnings about bool types

    # TODO - find and save projection type

    infile = os.path.expanduser(infile[:-1]) if infile.endswith('/') else os.path.expanduser(infile)
    infile = infile if infile.startswith('/') else './' + infile
    prefix = infile[:infile.rindex('.')]
    suffix = infile[infile.rindex('.') + 1:]
    srcdir = infile[:infile.rindex('/') + 1]
    if outfile == None: outfile = prefix + '.img.zarr'
    outfile = os.path.expanduser(outfile)

    if compressor is None:
        compressor = Blosc(cname='zstd', clevel=2, shuffle=0)

    tmp = os.system("rm -fr " + outfile)
    tmp = os.system("mkdir " + outfile)

    IA = ia()
    QA = qa()
    begin = time.time()
    
    # all image artifacts will go in same zarr file and share common dimensions if possible
    # check for meta data compatibility
    # store necessary coordinate conversion data
    if artifacts is None: artifacts = [suffix]
    elif len(artifacts) == 0: artifacts = ['image', 'pb', 'psf', 'residual', 'mask', 'model', 'sumwt', 'weight', 'image.pbcor']
    if suffix not in artifacts: artifacts = [suffix] + artifacts
    diftypes, mxds, artifact_dims, artifact_masks = [], xarray.Dataset(), {}, {}
    ttcount = 0

    # for each image artifact, determine what image files in the source directory are compatible with each other
    # extract the metadata from each
    # if taylor terms are present for the artifact, process metadata for first one only
    print("converting Image...")
    dirlist = sorted([srcdir+ff for ff in os.listdir(srcdir) if (srcdir+ff).startswith(prefix)])
    for imtype in artifacts:
        imagelist = [ff for ff in dirlist if len(re.findall('%s\.%s$'%(prefix, imtype), ff))>0]
        if len(imagelist)==0: imagelist = [ff for ff in dirlist if len(re.findall('%s\.%s\.tt\d+$'%(prefix, imtype), ff))>0]
        if (len(imagelist)==0) and (len(imtype.split('.'))>1):
            imagelist = [ff for ff in dirlist if len(re.findall('%s\.%s\.tt\d\.%s$'%(prefix, imtype.split('.')[0], imtype.split('.')[1]), ff))>0]
        if len(imagelist) == 0: continue

        # find number of taylor terms for this artifact and update count for total set if necessary
        ttcount = len(imagelist) if ttcount == 0 else ttcount

        rc = IA.open(imagelist[0])
        csys = IA.coordsys()
        summary = IA.summary(list=False)  # imhead would be better but chokes on big images
        ims = IA.shape()  # image shape
        coord_names = [ss.replace(' ', '_').lower().replace('stokes', 'pol').replace('frequency', 'chan') for ss in summary['axisnames']]
        
        # compute world coordinates for spherical dimensions
        sphr_dims = [dd for dd in range(len(ims)) if QA.isangle(summary['axisunits'][dd])]
        coord_idxs = np.mgrid[[range(ims[dd]) if dd in sphr_dims else range(1) for dd in range(len(ims))]].reshape(len(ims), -1)
        coord_world = csys.toworldmany(coord_idxs.astype(float))['numeric'][sphr_dims].reshape((-1,)+tuple(ims[sphr_dims]))
        coords = dict([(coord_names[dd], (['l','m'], coord_world[di])) for di, dd in enumerate(sphr_dims)])
        if imtype == 'sumwt': coords = {}   # special case, force sumwt to only cartesian coords (chan, pol)
        
        # compute world coordinates for cartesian dimensions
        cart_dims = [dd for dd in range(len(ims)) if dd not in sphr_dims]
        coord_idxs = np.mgrid[[range(ims[dd]) if dd in cart_dims else range(1) for dd in range(len(ims))]].reshape(len(ims), -1)
        coord_world = csys.toworldmany(coord_idxs.astype(float))['numeric'][cart_dims].reshape((-1,)+tuple(ims[cart_dims]))
        for dd, cs in enumerate(list(coord_world)):
            spi = tuple([slice(None) if di == dd else slice(1) for di in range(cs.ndim)])
            coords.update(dict([(coord_names[cart_dims[dd]], cs[spi].reshape(-1))]))

        # compute the time coordinate
        dtime = csys.torecord()['obsdate']['m0']['value']
        if csys.torecord()['obsdate']['m0']['unit'] == 'd': dtime = dtime * 86400
        coords['time'] = convert_time([dtime])

        # assign values to l, m coords based on incr and refpix in metadata
        if ('incr' in summary) and ('refpix' in summary) and ('shape' in summary) and (imtype != 'sumwt'):
            coords['l'] = np.arange(-summary['refpix'][0], summary['shape'][0]-summary['refpix'][0]) * summary['incr'][0]
            coords['m'] = np.arange(-summary['refpix'][1], summary['shape'][1]-summary['refpix'][1]) * summary['incr'][1]

        # check to see if this image artifact is of a compatible shape to be part of the image artifact dataset
        try:  # easiest to try to merge and let xarray figure it out
            mxds = mxds.merge(xarray.Dataset(coords=coords), compat='equals')
        except Exception:
            diftypes += [imtype]
            continue

        # store rest of image metadata as attributes (if not already in the xds
        omits = list(mxds.attrs.keys())
        omits += ['hasmask', 'masks', 'defaultmask', 'ndim', 'refpix', 'refval', 'shape', 'tileshape', 'messages']
        nested = [kk for kk in summary.keys() if isinstance(summary[kk], dict)]
        mxds = mxds.assign_attrs(dict([(kk.lower(), summary[kk]) for kk in summary.keys() if kk not in omits + nested]))
        artifact_dims[imtype] = [ss.replace('right_ascension', 'l').replace('declination', 'm') for ss in coord_names]
        artifact_masks[imtype] = summary['masks']
        
        # manually swap known meta attrs that deal with dim/coord units
        if ('axisnames' in summary) and ('axisnames' not in omits):
            mxds = mxds.assign_attrs({'axisnames':list(mxds.axisnames[:2]) + ['Time'] + list(mxds.axisnames[2:][::-1])})
        if ('axisunits' in summary) and ('axisunits' not in omits):
            mxds = mxds.assign_attrs({'axisunits':list(mxds.axisunits[:2]) + ['datetime64[ns]'] + list(mxds.axisunits[2:][::-1])})

        # check for common and restoring beams
        rb = IA.restoringbeam()
        if (len(rb) > 0) and ('restoringbeam' not in mxds.attrs):
            # if there is a restoring beam, this should work
            cb = IA.commonbeam()
            mxds = mxds.assign_attrs({'commonbeam': [cb['major']['value'], cb['minor']['value'], cb['pa']['value']]})
            mxds = mxds.assign_attrs({'commonbeam_units': [cb['major']['unit'], cb['minor']['unit'], cb['pa']['unit']]})
            mxds = mxds.assign_attrs({'restoringbeam': [cb['major']['value'], cb['minor']['value'], cb['pa']['value']]})
            if 'beams' in rb:
                beams = np.array([[rbs['major']['value'], rbs['minor']['value'], rbs['positionangle']['value']]
                                                          for rbc in rb['beams'].values() for rbs in rbc.values()])
                mxds = mxds.assign_attrs({'perplanebeams':beams.reshape(len(rb['beams']),-1,3)})
                
        # parse messages for additional keys, drop duplicate info
        omits = list(mxds.attrs.keys()) + ['image_name', 'image_type', 'image_quantity', 'pixel_mask(s)', 'region(s)', 'image_units']
        for msg in summary['messages']:
            line = [tuple(kk.split(':')) for kk in msg.lower().split('\n') if ': ' in kk]
            line = [(kk[0].strip().replace(' ', '_'), kk[1].strip()) for kk in line]
            line = [ll for ll in line if ll[0] not in omits]
            mxds = mxds.assign_attrs(dict(line))
                
        rc = csys.done()
        rc = IA.close()

    if len(diftypes) > 0: print('incompatible components: ', diftypes)

    # if taylor terms are present, the chan axis must be expanded to the length of the terms
    if ttcount > len(mxds.chan): mxds = mxds.pad({'chan': (0, ttcount-len(mxds.chan))}, mode='edge')
   
    
    chunk_dict = dict(zip(['l','m','time','chan','pol'], chunks[:2]+(1,)+chunks[2:]))
    mxds = mxds.chunk(chunk_dict)

    # for each artifact, convert the legacy format and add to the new image set
    # masks may be stored within each image, so they will need to be handled like subtables
    for ac, imtype in enumerate(list(artifact_dims.keys())):
        for ec, ext in enumerate([''] + ['/'+ff for ff in list(artifact_masks[imtype])]):
            imagelist = [ff for ff in dirlist if len(re.findall('%s\.%s$' % (prefix, imtype), ff)) > 0]
            if len(imagelist) == 0: imagelist = [ff for ff in dirlist if len(re.findall('%s\.%s\.tt\d+$' % (prefix, imtype), ff)) > 0]
            if (len(imagelist) == 0) and (len(imtype.split('.')) > 1):
                imagelist = [ff for ff in dirlist if len(re.findall('%s\.%s\.tt\d\.%s$' % (prefix, imtype.split('.')[0], imtype.split('.')[1]), ff)) > 0]
            if len(imagelist) == 0: continue

            dimorder = ['time'] + list(reversed(artifact_dims[imtype]))
            chunkorder = [chunk_dict[vv] for vv in dimorder]
            ixds = convert_simple_table(imagelist[0]+ext, outfile+'.temp', dimnames=dimorder, compressor=compressor, chunks=chunkorder)

            # if the image set has taylor terms, loop through any for this artifact and concat together
            # pad the chan dim as necessary to fill remaining elements if not enough taylor terms in this artifact
            for ii in range(1, ttcount):
                if (ii < len(imagelist)) and os.path.exists(imagelist[ii] + ext):
                    txds = convert_simple_table(imagelist[ii]+ext, outfile + '.temp', dimnames=dimorder, chunks=chunkorder, nofile=True)
                    ixds = xarray.concat([ixds, txds], dim='chan')
                else:
                    ixds = ixds.pad({'chan': (0, 1)}, constant_values=np.nan)

            ixds = ixds.rename({list(ixds.data_vars)[0]:(imtype.replace('.','_')+ext.replace('/','_')).upper()}).transpose('l','m','time','chan','pol')
            if imtype == 'sumwt': ixds = ixds.squeeze(['l','m'], drop=True)
            if imtype == 'mask': ixds = ixds.rename({'MASK':'AUTOMASK'})  # rename mask

            encoding = dict(zip(list(ixds.data_vars), cycle([{'compressor': compressor}])))
            ixds.to_zarr(outfile, mode='w' if (ac==0) and (ec==0) else 'a', encoding=encoding, compute=True, consolidated=True)

    tmp = os.system("rm -fr " + outfile+'.temp')
    print('processed image in %s seconds' % str(np.float32(time.time() - begin)))

    # add attributes from metadata and version tag file
    mxds.to_zarr(outfile, mode='a', compute=True, consolidated=True)
    try:
        version = importlib_metadata.version('cngi-prototype')
    except:
        version = '0.0.0'

    with open(outfile + '/.version', 'w') as fid:   # write sw version that did this conversion to zarr directory
        fid.write('cngi-protoype ' + version + '\n')

    return xarray.open_zarr(outfile)
