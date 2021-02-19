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
"""
this module will be included in the api
"""

#############################################
def read_vis(infile, partition=None, chunks=None, consolidated=True, overwrite_encoded_chunks=True, **kwargs):
    """
    Read zarr format Visibility data from disk to xarray Dataset

    Parameters
    ----------
    infile : str
        input Visibility filename
    partition : string or list
        name of partition(s) to read as returned by describe_vis. Multiple partitions in list form will return a master dataset of datasets.
        Use 'global' for global metadata. Default None returns everything
    chunks : dict
        sets specified chunk size per dimension. Dict is in the form of 'dim':chunk_size, for example {'time':100, 'baseline':400, 'chan':32, 'pol':1}.
        Default None uses the original zarr chunking.
    consolidated : bool
        use zarr consolidated metadata capability. Only works for stores that have already been consolidated. Default True works with datasets
        produced by convert_ms which automatically consolidates metadata.
    overwrite_encoded_chunks : bool
        drop the zarr chunks encoded for each variable when a dataset is loaded with specified chunk sizes.  Default True, only applies when chunks
        is not None.
    s3_key : string, optional
        optional support for explicit authentication if infile is provided as S3 URL. 
        If S3 url is passed as input but this argument is not specified then only publicly-available, read-only buckets are accessible (so output dataset will be read-only).
    s3_secret : string, optional
        optional support for explicit authentication if infile is provided as S3 URL. 
        If S3 url is passed as input but this argument is not specified then only publicly-available, read-only buckets are accessible (so output dataset will be read-only).

    Returns
    -------
    xarray.core.dataset.Dataset
        New xarray Dataset of Visibility data contents
    """
    import os
    import numpy as np
    import cngi._utils._io as xdsio
    from xarray import open_zarr

    if infile.lower().startswith('s3'):

        # for treating AWS object storage as a "file system"
        import s3fs

        if 's3_key' and 's3_secret' in kwargs:
            # plaintext authentication is a security hazard that must be patched ASAP
            # boto3 can be used instead, see https://s3fs.readthedocs.io/en/latest/#credentials
            # if we instead choose to extend the current solution, might want to santiize inputs
            s3 = s3fs.S3FileSystem(anon=False, requester_pays=False, key=kwargs['s3_key'], secret=kwargs['myvalues'])

        else:
            # only publicly-available, read-only buckets will work. Should probably catch the exception here...
            s3 = s3fs.S3FileSystem(anon=True, requester_pays=False)

        # expect a path style URI to file link, e.g.,
        # 's3://cngi-prototype-test-data/2017.1.00271.S/member.uid___A001_X1273_X2e3_split_cal_concat_target_regrid.vis.zarr/xds0/'
        # decompose this for manipulation
        s3_url = infile.split(sep='//', maxsplit=1)[1]
        bucket = s3_url.split('/')[0]
        name = s3_url.split('/')[1]
        partition = s3_url.split('/')[2]
        ds_path = '/'.join([bucket,name])

        if partition is not '':
            if s3.isdir(ds_path):
                object_list = s3.listdir(ds_path)
                partition = [ol['name'].split('/')[-1] for ol in object_list[1:]]
                
        if chunks is None:
            chunks = 'auto'
            overwrite_encoded_chunks = False

        if ('global' in partition) and s3.isdir('/'.join([ds_path,'global'])):
            object_list_global = s3.listdir('/'.join([ds_path,'global']))
            # attempt to match behavior of os.listdir (i.e., ignore .zattrs et cetera
            olg_dirs = [ol['name'].split('/')[-1] for ol in object_list_global[1:] if ol['StorageClass'] == 'DIRECTORY']
            global_dirs = sorted(['global/'+tt for tt in olg_dirs])
            partition = np.hstack((np.delete(partition, np.where(partition == 'global')), global_dirs))

        if partition.size == 1:
            # first convert the S3 key:value objects to a MutableMapping
            INPUT = s3fs.S3Map(root='/'.join([bucket,vis_data,partition]), s3=s3, check=False)
            # then pass the mapping to xarray's zarr interface as normal
            xds = open_zarr(INPUT, chunks=chunks, consolidated=consolidated, overwrite_encoded_chunks=overwrite_encoded_chunks)
        else:
            xds_list = []
            for part in partition:
                if s3.isdir('/'.join([ds_path,str(part)])):
                    INPUT = s3fs.S3Map(root='/'.join([ds_path,str(part)]), s3=s3, check=False)
                    xds_list += [(part.replace('global/',''), open_zarr(INPUT, chunks=chunks, consolidated=consolidated, 
                                                                        overwrite_encoded_chunks=overwrite_encoded_chunks))]
    else: # the non-s3 case, access data via local filesystem
        infile = os.path.expanduser(infile)
        if partition is None: partition = os.listdir(infile)
        partition = np.atleast_1d(partition)

        if chunks is None:
            chunks = 'auto'
            overwrite_encoded_chunks = False

        if ('global' in partition) and (os.path.isdir(os.path.join(infile,'global'))):
            global_dirs = sorted(['global/'+tt for tt in os.listdir(os.path.join(infile,'global'))])
            partition = np.hstack((np.delete(partition, np.where(partition == 'global')), global_dirs))
  
        if partition.size == 1:
            xds = open_zarr(os.path.join(infile, str(partition[0])), chunks=chunks, consolidated=consolidated,
                            overwrite_encoded_chunks=overwrite_encoded_chunks)
        else:
            xds_list = []
            for part in partition:
                if os.path.isdir(os.path.join(infile, str(part))):
                    xds_list += [(part.replace('global/',''), open_zarr(os.path.join(infile, str(part)), chunks=chunks, consolidated=consolidated,
                                                                        overwrite_encoded_chunks=overwrite_encoded_chunks))]
    # build the master xds to return
    xds = xdsio.vis_xds_packager(xds_list)

    return xds

