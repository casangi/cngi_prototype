"""
Flagging subpackage modules

These functions can be used to calculate flags and handle different flag versions.
There are functions to inspect flag variables and manage the set of flag variables
in a CNGI Dataset. Other functions calculate flags using different flagging
methods.
"""
from .auto_clip import auto_clip
from .auto_rflag import auto_rflag
from .auto_tfcrop import auto_tfcrop
from .auto_uvbin import auto_uvbin
from .elevation import elevation
from .extend import extend
from .manager_add import manager_add
from .manager_list import manager_list
from .manager_remove import manager_remove
from .manual_flag import manual_flag
from .manual_unflag import manual_unflag
from .quack import quack
from .shadow import shadow
from .summary import summary
