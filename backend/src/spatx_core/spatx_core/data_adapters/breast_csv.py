import os

import pandas as pd

from .base_data_adapter import BaseDataAdapter
from ..data import DataPoint, PredictionDataPoint

class BreastDataAdapter(BaseDataAdapter):
    name = 'Breast Data Adapter'
    
    def __init__(self, image_dir : str, breast_csv : str, wsi_ids : list[str], gene_ids : list[str]):
        if not os.path.exists(image_dir):
            raise ValueError(f"The path {image_dir} does not exists")
        if not os.path.exists(breast_csv):
            raise FileNotFoundError(f"The file {breast_csv} does not exists")
        self.image_dir = image_dir
        self.breast_csv = breast_csv
        self.df : pd.DataFrame = pd.read_csv(self.breast_csv) #type: ignore
        self.wsi_ids = wsi_ids
        missing_wsi_ids = [wsi for wsi in self.wsi_ids if wsi not in self.df['id'].unique()]
        if missing_wsi_ids:
            raise ValueError(f"The following WSI IDs are missing in the CSV: {missing_wsi_ids}")
        metadata_cols = ['barcode', 'id', 'x_pixel', 'y_pixel', 'combined_text']
        self.gene_ids = sorted(gene_ids)
        if len(gene_ids) != len(set(gene_ids)):
            duplicates = [gene for gene in gene_ids if gene_ids.count(gene) > 1]
            raise ValueError(f"Duplicate gene IDs found: {duplicates}")
        self.df = self.df[self.df['id'].isin(self.wsi_ids)] #type: ignore
        missing = [col for col in gene_ids if col not in self.df.columns]
        if missing:
            raise ValueError(f"The following columns are missing in DataFrame: {missing}")
        self.df = self.df[metadata_cols+self.gene_ids]
        self.gene_cols = [col for col in self.df.columns if col not in metadata_cols]
        

    def __getitem__(self, idx : int):
        row = self.df.iloc[idx]
        barcode  = row['barcode']
        wsi_id   = row['id']
        img_patch_name = f"{barcode}_{wsi_id}.png"
        img_patch_path = os.path.join(self.image_dir, img_patch_name)
        if not os.path.exists(img_patch_path):
            raise FileNotFoundError(f"Image patch not found: {img_patch_path}")

        
        # Create gene expression dictionary
        gene_expression = {gene: float(row[gene]) for gene in self.gene_cols}
        
        return DataPoint(
            x = row['x_pixel'],
            y = row['y_pixel'],
            img_patch_path= img_patch_path,
            gene_expression= gene_expression,
            wsi_id= wsi_id,
            barcode= barcode,
            )

    def __len__(self):
        return len(self.df)
    
class BreastPredictionDataAdapter(BaseDataAdapter):
    name = 'Breast Prediction Data Adapter'
    
    def __init__(self, image_dir: str, prediction_csv: str, wsi_ids: list[str]):
        if not os.path.exists(image_dir):
            raise ValueError(f"The path {image_dir} does not exist")
        if not os.path.exists(prediction_csv):
            raise FileNotFoundError(f"The file {prediction_csv} does not exist")
        self.image_dir = image_dir
        self.prediction_csv = prediction_csv
        self.df: pd.DataFrame = pd.read_csv(self.prediction_csv) #type: ignore
        self.wsi_ids = wsi_ids
        
        # Check required columns
        required_cols = ['barcode', 'id', 'x_pixel', 'y_pixel']
        missing = [col for col in required_cols if col not in self.df.columns]
        if missing:
            raise ValueError(f"The following required columns are missing in DataFrame: {missing}")
        
        missing_wsi_ids = [wsi for wsi in self.wsi_ids if wsi not in self.df['id'].unique()]
        if missing_wsi_ids:
            raise ValueError(f"The following WSI IDs are missing in the CSV: {missing_wsi_ids}")
        
        # Filter dataframe to only include specified WSI IDs
        self.df = self.df[self.df['id'].isin(self.wsi_ids)] #type: ignore

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        barcode = row['barcode']
        wsi_id = row['id']
        img_patch_name = f"{barcode}_{wsi_id}.png"
        img_patch_path = os.path.join(self.image_dir, img_patch_name)
        if not os.path.exists(img_patch_path):
            raise FileNotFoundError(f"Image patch not found: {img_patch_path}")
        
        return PredictionDataPoint(
            x=row['x_pixel'],
            y=row['y_pixel'],
            img_patch_path=img_patch_path,
            wsi_id=wsi_id,
            barcode=barcode,
        )

    def __len__(self):
        return len(self.df)