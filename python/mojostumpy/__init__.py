"""Matrix-profile computation accelerated with Mojo."""

from .core import aamp, aampdist, mass, match, mpdist, stump
from .mparray import mparray

__all__ = ["aamp", "aampdist", "mass", "match", "mpdist", "mparray", "stump"]
__version__ = "0.1.0"
