# C
"""
this module will be included in the api
"""
import pandas as pd
from ._flagging_utils._flag_attributes import _ensure_flags_attr


def manager_list(vis_dataset):
    """
    Add a new flag variable to the dataset. All flags in the new variable are
    set to false, unless a source variable is given in which case the values of
    the source are copied.

    Parameters
    ----------
    vis_dataset : xarray.core.dataset.Dataset
        Input dataset

    Returns:
    -------
    pandas.core.frame.DataFrame
        Information on flag variables from the input dataset
    """
    flags_attr = _ensure_flags_attr(vis_dataset)

    res = {'Flag variable name':
           [key for key in vis_dataset.attrs[flags_attr]],
           'Description':
           [val for _key, val in vis_dataset.attrs[flags_attr].items()]}
    return pd.DataFrame.from_dict(res)
