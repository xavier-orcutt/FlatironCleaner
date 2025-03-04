import pandas as pd
import numpy as np
import logging
import math 
import re 
from typing import Optional

logging.basicConfig(
    level = logging.INFO,                                 
    format = '%(asctime)s - %(levelname)s - %(message)s'  
)

class DataProcessorGeneral:
    
    def __init__(self):