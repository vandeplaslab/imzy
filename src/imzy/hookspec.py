"""Hooks for imzy."""
import typing as ty

from koyo.typing import PathLike
from pluggy import HookimplMarker, HookspecMarker

if ty.TYPE_CHECKING:
    from imzy._readers import BaseReader

hook_spec: ty.Callable = HookspecMarker("imzy")
hook_impl: ty.Callable = HookimplMarker("imzy")


@hook_spec(firstresult=False)
def imzy_reader(path: PathLike, **kwargs) -> ty.Optional["BaseReader"]:
    """Hook specification for file reader.

    This function should return instance of initialized reader if the path is supported by the reader. You should
    implement the following login in the function:

    1. Check whether the specified 'path' is supported by the reader.
    2. Make any or all checks that ensures that the path is what you think it is.
    3. If the path is supported, return instance of the reader, otherwise return None.

    See examples of `imzy._readers` for more information.

    Parameters
    ----------
    path : str
        Path to the dataset.
    kwargs : dict
        Dictionary of extra keyword-arguments that should be passed to the reader. For definition of what arguments are
        supported, please see individual readers.

    Returns
    -------
    reader : BaseReader
        Reader object if the path is supported, otherwise None.
    """
    ...
