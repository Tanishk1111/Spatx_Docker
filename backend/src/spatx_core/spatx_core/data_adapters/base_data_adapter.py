from __future__ import annotations
from typing import TYPE_CHECKING
from abc import ABC, abstractmethod

if TYPE_CHECKING:
    from ..data import DataPoint

class BaseDataAdapter(ABC):
    name = 'Unnamed Adapter'

    @abstractmethod
    def __init__(self, gene_ids: list[str], *args, **kwargs): # type: ignore
        pass
    
    @abstractmethod
    def __getitem__(self, idx : int) -> DataPoint:
        '''
        Should be implemented to return a data point in a format the the standard Data class requires
        '''
        pass
    
    @abstractmethod
    def __len__(self) -> int:
        '''
        Should be implemented to return the length of data in a format the the standard Data class requires
        '''
        pass