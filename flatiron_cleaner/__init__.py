"""
FlatironCleaner

A Python package for cleaning and harmonizing Flatiron Health cancer data
structured around specified index dates of interest.
"""

__version__ = '0.1.0'

# Make key classes available at package level
from .urothelial import DataProcessorUrothelial
from .nsclc import DataProcessorNSCLC
from .colorectal import DataProcessorColorectal
from .breast import DataProcessorBreast
from .prostate import DataProcessorProstate
from .general import DataProcessorGeneral
from .merge_utils import merge_dataframes

# Define what gets imported with `from flatiron_cleaner import *`
__all__ = [
    'DataProcessorUrothelial',
    'DataProcessorNSCLC',
    'DataProcessorColorectal',
    'DataProcessorBreast',
    'DataProcessorProstate',
    'DataProcessorGeneral',
    'merge_dataframes',
]