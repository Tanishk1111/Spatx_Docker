import warnings
import os

class DataPoint:
    def __init__(self, x : int, y : int, img_patch_path : str, gene_expression : dict[str, float], wsi_id : str | None = None, barcode : str | None = None):
        self.x = x
        self.y = y
        self.img_patch_path = img_patch_path
        self.gene_expression = gene_expression # dict of gene id, float
        self.wsi_id = wsi_id
        self.barcode = barcode

    def validate_datapoint(self, adapter_name : str):
        if self.wsi_id is None:
            warnings.warn(f"The Adapter {adapter_name} is giving a datapoint without wsi id, , if this is intentional please ignore this warning")
        if self.barcode is None:
            warnings.warn(f"The Adapter {adapter_name} is giving a datapoint without barcode, if this is intentional please ignore this warning")
        if not os.path.exists(self.img_patch_path):
            raise ValueError(f"There is no image at path provided by adapter {adapter_name} at {self.img_patch_path}")
        if not isinstance(self.gene_expression, dict): # type: ignore
            raise ValueError(f"The adapter {adapter_name} is not returning a gene_expression object in format dict[str, float] instead returning in format {type(self.gene_expression)}")
        if not isinstance(self.x, float | int) or not isinstance(self.y, float | int): # type: ignore
            warnings.warn(f"The Adapter {adapter_name} is returning x and y which are not instances of int | float class, instead x is instance of {type(self.x)} and y is instance of {type(self.y)}, if this is intentional please ignore this warning")
        return True

class PredictionDataPoint:
    def __init__(self, x: int, y: int, img_patch_path: str, wsi_id: str | None = None, barcode: str | None = None):
        self.x = x
        self.y = y
        self.img_patch_path = img_patch_path
        self.wsi_id = wsi_id
        self.barcode = barcode

    def validate_datapoint(self, adapter_name: str):
        if self.wsi_id is None:
            warnings.warn(f"The Adapter {adapter_name} is giving a datapoint without wsi id, if this is intentional please ignore this warning")
        if self.barcode is None:
            warnings.warn(f"The Adapter {adapter_name} is giving a datapoint without barcode, if this is intentional please ignore this warning")
        if not os.path.exists(self.img_patch_path):
            raise ValueError(f"There is no image at path provided by adapter {adapter_name} at {self.img_patch_path}")
        if not isinstance(self.x, float | int) or not isinstance(self.y, float | int): # type: ignore
            warnings.warn(f"The Adapter {adapter_name} is returning x and y which are not instances of int | float class, instead x is instance of {type(self.x)} and y is instance of {type(self.y)}, if this is intentional please ignore this warning")
        return True
