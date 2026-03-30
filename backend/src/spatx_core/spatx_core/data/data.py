from __future__ import annotations
import warnings
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..data_adapters.base_data_adapter import BaseDataAdapter
from .data_point import DataPoint, PredictionDataPoint

class Data:
    '''
    This is the standard data class that standardizes the data format of spatial transcriptomics data, for any model
    classes which inherit from torch.utils.data.Dataset can use this class to retrieve data in a standard format, and provide it to the model in a format that the model expects.
    The valid data format expected from the Adapter is:
        length : an integer
        data point dictionary:
            - a image patch path
            - x coordinate
            - y coordinate
            - arr[gene_id, expression level]
    '''
    def __init__(self, adapter: BaseDataAdapter):
        self.adapter = adapter
        if not self._validate_adapter():
            raise ValueError(f"The adapter : {self.adapter.name} is not returning a standard data point")
    
    def __getitem__(self, idx : int) -> DataPoint:
        # Delegate to adapter for actual data loading
        return self.adapter[idx]
    
    def __len__(self):
        return len(self.adapter)
    

    def _validate_adapter(self):
        '''
        Function to check weather the adapter's __get_item__ returns the data in a format which is standardised here
        '''
        if len(self.adapter) > 0:
             if isinstance(self.adapter[0], DataPoint): # type: ignore
                return self.adapter[0].validate_datapoint(adapter_name= self.adapter.name)
             return False
        raise RuntimeError(f"The Current Prediction Data adapter: {self.adapter.name} is providing zero data length")
        
class PredictionData:
    '''
    This is the prediction data class that standardizes the data format of spatial transcriptomics data for prediction,
    for dataset classes which inherit from torch.utils.data.Dataset can use this class to retrieve data in a standard format.
    The valid data format expected from the Adapter is:
        length : an integer
        prediction data point:
            - a image patch path
            - x coordinate
            - y coordinate
    '''
    def __init__(self, adapter: BaseDataAdapter):
        self.adapter = adapter
        if not self._validate_adapter():
            raise ValueError(f"The adapter : {self.adapter.name} is not returning a standard prediction data point")
    
    def __getitem__(self, idx: int) -> PredictionDataPoint:
        # Delegate to adapter for actual data loading
        return self.adapter[idx]
    
    def __len__(self):
        return len(self.adapter)
    
    def _validate_adapter(self):
        '''
        Function to check whether the adapter's __get_item__ returns the data in a format which is standardised here
        '''
        if len(self.adapter) > 0:
             if isinstance(self.adapter[0], PredictionDataPoint): # type: ignore
                return self.adapter[0].validate_datapoint(adapter_name=self.adapter.name)
             return False
        raise RuntimeError(f"The Current Prediction Data adapter: {self.adapter.name} is providing zero data length")
        