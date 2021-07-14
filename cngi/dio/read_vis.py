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

#############################################
def read_vis(
    infile,
    partition=None,
    chunks=None,
    consolidated=True,
    overwrite_encoded_chunks=True,
    **kwargs,
):
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

    if chunks is None:
        chunks = "auto"
        #overwrite_encoded_chunks = False
    print('overwrite_encoded_chunks',overwrite_encoded_chunks)

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
                secret=kwargs["s3_secret"],
            )

        else:
            # only publicly-available, read-only buckets will work. Should probably catch the exception here...
            s3 = s3fs.S3FileSystem(anon=True, requester_pays=False)

        # expect a path style URI to file link, e.g.,
        # 's3://cngi-prototype-test-data/2017.1.00271.S/member.uid___A001_X1273_X2e3_split_cal_concat_target_regrid.vis.zarr/xds0/'
        s3_url = infile.split(sep="//", maxsplit=1)[1]
        # trim trailing slashes
        while s3_url.endswith("/"):
            s3_url = s3_url[:-1]

        if s3.isdir(s3_url):
            # this conditional is first otherwise there's no point to continue
            contents_map = s3.listdir(s3_url)
            object_names = [
                object_dict["name"].split("/")[-1] for object_dict in contents_map
            ]
            if "time" and "baseline" and "chan" and "pol" in object_names:
                # looks like the input URI was one level too deep or s3_url points to a pre-0.0.65 xds *shivers*
                if partition is None:
                    partition = s3_url.split("/")[-1]
                if partition != s3_url.split("/")[-1]:
                    # includes case of empty partition kwarg but included in infile string
                    # we should agree on doing something more solid here
                    # e.g., isinstance(partition, str) and isinstance(partition, list)
                    print(
                        "Input to partition keyword argument does not match provided S3 URI"
                    )
                    partition = s3_url.split("/")[-1]
                    print(f"Assigning partition = {partition}")
                    s3_url = ("/").join(s3_url.split("/")[:-1])

            # at this point, s3_url should be compatible and reference top level of a mxds
            if partition is None:
                 # avoid the .version object
                contents_map = s3.listdir(s3_url)[1:]
                object_names = [
                    object_dict["name"].split("/")[-1] for object_dict in contents_map
                ]
                object_names = [
                    oname for oname in object_names if not oname.startswith(".")
                ]
                partition = object_names
                
            if "global" in partition:
                # attempt to replicate behavior of os.listdir (i.e., ignore .zattrs etc.)
                contents_map_global = s3.listdir("/".join([s3_url, "global"]))
                olg_dirs = [
                    odg["name"].split("/")[-1]
                    for odg in contents_map_global
                    if odg["StorageClass"] == "DIRECTORY"
                ]
                global_dirs = sorted(["global/" + od for od in olg_dirs])
                if isinstance(partition, list):
                    partition.remove("global")
                    partition = np.asarray(partition + global_dirs)
                else:
                    partition = np.hstack(
                        (
                            np.delete(partition, np.where(partition == "global")),
                            global_dirs,
                        )
                    )

            # now ready to read
            xds_list = []
            if isinstance(partition, np.ndarray):
                for part in partition:
                    uri = "/".join([s3_url, str(part)])
                    if s3.isdir(uri):
                        INPUT = s3fs.S3Map(root=uri, s3=s3, check=False)
                        xds_list += [
                            (
                                uri.replace(s3_url + "/", "").replace("global/", ""),
                                open_zarr(
                                    INPUT,
                                    chunks=chunks,
                                    consolidated=consolidated,
                                    overwrite_encoded_chunks=overwrite_encoded_chunks,
                                ),
                            )
                        ]
                    else:
                        print(f"Requested partition {part} not found in dataset")
            else:
                # this case should hit only for single str input (unencased by list) to partition kwarg
                uri = "/".join([s3_url, partition])
                INPUT = s3fs.S3Map(root=uri, s3=s3, check=False)
                xds = open_zarr(
                    INPUT,
                    chunks=chunks,
                    consolidated=consolidated,
                    overwrite_encoded_chunks=overwrite_encoded_chunks,
                )
                xds_list.append(xds)

    else:  # the non-s3 case, access data via local filesystem
        infile = os.path.expanduser(infile)
        if partition is None:
            partition = os.listdir(infile)
        partition = list(np.atleast_1d(partition))

        if ("global" in partition) and (os.path.isdir(os.path.join(infile, "global"))):
            partition += sorted(["global/"+tt for tt in os.listdir(os.path.join(infile, "global"))])

        xds_list = []
        for part in partition:
            if part == 'global': continue
            try:
                if os.path.isdir(os.path.join(infile, str(part))):
                    xds_list += [(part.replace("global/", ""), open_zarr(os.path.join(infile, str(part)), chunks=chunks,
                                                                         consolidated=consolidated,
                                                                         overwrite_encoded_chunks=overwrite_encoded_chunks))]
            except:
                print('Can not open ', part)
    # build the master xds to return
    xds = xdsio.vis_xds_packager(xds_list)

    return xds
