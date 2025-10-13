from __future__ import annotations from abc import ABC, abstractmethod 
from abc import ABC, abstractmethod 
from typing import Dict, List, Tuple, Iterable, Optional 
import numpy as np


class BenchMark(ABC): 

    @abstractmethod 
    def update(seld, *args, **kwargs):
        "Pass bacth of data or portion of that, for instance a frame"
        ... 

    @abstractmethod 
    def finalize(self) -> None: 
        "Compute metrics from accumulated state" 
        ...  

    
    @abstractmethod 
    def results(self) -> Dict[str, float]: 
        "Gather computed metrics as a flat dict" 
        ... 

    @abstractmethod
    def reset(self)->None: 
        "Clear internal state" 
        ... 





