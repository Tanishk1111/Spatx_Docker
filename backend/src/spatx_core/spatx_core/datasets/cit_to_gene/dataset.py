import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms #type: ignore
from PIL import Image

from ...data import Data, PredictionData

class CITDataset(Dataset):
    '''
    This class is for providing data to the CitToGene model.
    
    Returns:
        tuple: (image_tensor, expression_tensor, sample_id)
            - image_tensor: torch.Tensor of shape [3, 224, 224], normalized with ImageNet stats
            - expression_tensor: torch.Tensor of shape [num_genes], containing gene expression values
            - sample_id: str, unique identifier for the sample
    '''
    def __init__(self, data: Data):
        self.data = data
        # Define standard image transformations for CIT model
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet stats
                std=[0.229, 0.224, 0.225]
            )
        ])

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx : int):
        """
        Get a single sample from the dataset.
        
        Args:
            idx (int): Index of the sample to fetch
            
        Returns:
            tuple: (image_tensor, expression_tensor, sample_id)
        """
        # Get data point from the Data class
        datapoint = self.data[idx]
        
        # Load and transform image
        image = Image.open(datapoint.img_patch_path).convert('RGB')
        image_tensor = self.transform(image)
        
        # Convert gene expression dictionary to tensor
        # Ensure consistent gene ordering by sorting keys
        gene_names = sorted(datapoint.gene_expression.keys())
        expression_values = [datapoint.gene_expression[gene] for gene in gene_names]
        expression_tensor = torch.tensor(expression_values, dtype=torch.float32)
        
        # Create sample ID from coordinates
        sample_id = f"{datapoint.barcode}_{datapoint.wsi_id}"
        return image_tensor, expression_tensor, sample_id


class CITPredictionDataset(Dataset):
    '''
    This class is for providing prediction data to the CitToGene model.
    
    Returns:
        tuple: (image_tensor, sample_info)
            - image_tensor: torch.Tensor of shape [3, 224, 224], normalized with ImageNet stats
            - sample_info: dict containing {'image_name': str, 'x': int, 'y': int, 'barcode': str, 'wsi_id': str}
    '''
    def __init__(self, prediction_data: PredictionData):
        self.prediction_data = prediction_data
        # Define standard image transformations for CIT model
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet stats
                std=[0.229, 0.224, 0.225]
            )
        ])

    def __len__(self):
        return len(self.prediction_data)
    
    def __getitem__(self, idx: int):
        """
        Get a single sample from the dataset.
        
        Args:
            idx (int): Index of the sample to fetch
            
        Returns:
            tuple: (image_tensor, sample_info)
        """
        # Get data point from the PredictionData class
        datapoint = self.prediction_data[idx]
        
        # Load and transform image
        image = Image.open(datapoint.img_patch_path).convert('RGB')
        image_tensor = self.transform(image)
        
        # Extract image name from path
        image_name = os.path.basename(datapoint.img_patch_path)
        
        # Create sample info dictionary
        sample_info = {
            'image_name': image_name,
            'x': datapoint.x,
            'y': datapoint.y,
            'barcode': datapoint.barcode,
            'wsi_id': datapoint.wsi_id
        }
        
        return image_tensor, sample_info