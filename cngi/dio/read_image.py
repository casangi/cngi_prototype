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
def read_image(
    infile, chunks=None, consolidated=True, overwrite_encoded_chunks=True, **kwargs
):
    """
    Read xarray zarr format image from disk

    Parameters
    ----------
    infile : str
        input zarr image filename
    chunks : dict
        sets specified chunk size per dimension. Dict is in the form of 'dim':chunk_size, for example {'d0':100, 'd1':100, 'chan':32, 'pol':1}.
        Default None uses the original zarr chunking.
    consolidated : bool
        use zarr consolidated metadata capability. Only works for stores that have already been consolidated. Default True works with datasets
        produced by convert_image which automatically consolidates metadata.
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
        New xarray Dataset of image contents
    """
    import os
    from xarray import open_zarr

    if chunks is None:
        chunks = "auto"
        overwrite_encoded_chunks = False

    if infile.lower().startswith("s3"):
        # for treating AWS object storage as a "file system"
        import s3fs

        if "s3_key" and "s3_secret" in kwargs:
            # plaintext authentication is a security hazard that must be patched ASAP
            # boto3 can be used instead, see https://s3fs.readthedocs.io/en/latest/#credentials
            # if we instead choose to extend the current solution, might want to santiize inputs
            s3 = s3fs.S3FileSystem(
                anon=False,
                requester_pays=False,
                key=kwargs["s3_key"],
                secret=kwargs["myvalues"],
            )

        else:
            # only publicly-available, read-only buckets will work. Should probably catch the exception here...
            s3 = s3fs.S3FileSystem(anon=True, requester_pays=False)

        # expect a path style URI to file link, e.g.,
        # 's3://cngi-prototype-test-data/2017.1.00271.S/member.uid___A001_X1273_X2e3_split_cal_concat_target_regrid.vis.zarr/xds0/'
        # decompose this for manipulation
        s3_url = infile.split(sep="//", maxsplit=1)[1]
        bucket = s3_url.split("/")[0]
        name = s3_url.split("/")[1]
        ds_path = "/".join([bucket, name])

        INPUT = s3fs.S3Map(root="/" + ds_path, s3=s3, check=False)
        xds = open_zarr(
            INPUT,
            chunks=chunks,
            consolidated=consolidated,
            overwrite_encoded_chunks=overwrite_encoded_chunks,
        )

    else:
        # assume infile exists on local disk
        infile = os.path.expanduser(infile)
        xds = open_zarr(
            infile,
            chunks=chunks,
            consolidated=consolidated,
            overwrite_encoded_chunks=overwrite_encoded_chunks,
        )

    return xds
