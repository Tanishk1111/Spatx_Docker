"""Tests for CITPredictionDataset class."""
import pytest
import torch
import numpy as np
from PIL import Image
import os

from spatx_core.datasets.cit_to_gene.dataset import CITPredictionDataset
from spatx_core.data.data import PredictionData
from spatx_core.data.data_point import PredictionDataPoint
from spatx_core.data_adapters.base_data_adapter import BaseDataAdapter


class MockCITPredictionDataAdapter(BaseDataAdapter):
    """Mock prediction data adapter for CITPredictionDataset testing."""
    
    name = "MockCITPredictionDataAdapter"
    
    def __init__(self, gene_ids, data_points=None):
        self.gene_ids = gene_ids  # Not used for prediction data but required by interface
        self.data_points = data_points or []
    
    def __getitem__(self, idx):
        return self.data_points[idx]
    
    def __len__(self):
        return len(self.data_points)


class TestCITPredictionDataset:
    """Test cases for CITPredictionDataset class."""
    
    def test_initialization_success(self, temp_dir):
        """Test successful CITPredictionDataset initialization."""
        # Create test image
        img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        img_path = os.path.join(temp_dir, "test_image.png")
        img.save(img_path)
        
        # Create prediction data point
        datapoint = PredictionDataPoint(
            x=100,
            y=200,
            img_patch_path=img_path,
            wsi_id="WSI001",
            barcode="ABC123"
        )
        
        # Create adapter and prediction data
        adapter = MockCITPredictionDataAdapter(
            gene_ids=[],  # No genes needed for prediction
            data_points=[datapoint]
        )
        prediction_data = PredictionData(adapter)
        
        # Initialize dataset
        dataset = CITPredictionDataset(prediction_data)
        
        assert dataset is not None
        assert len(dataset) == 1
        assert dataset.prediction_data == prediction_data
        assert dataset.transform is not None
    
    def test_getitem_success(self, temp_dir):
        """Test successful data retrieval via __getitem__."""
        # Create test image
        img = Image.fromarray(np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8))
        img_path = os.path.join(temp_dir, "test_prediction_image.png")
        img.save(img_path)
        
        # Create prediction data point
        datapoint = PredictionDataPoint(
            x=150,
            y=250,
            img_patch_path=img_path,
            wsi_id="WSI001",
            barcode="ABC123"
        )
        
        # Create adapter and prediction data
        adapter = MockCITPredictionDataAdapter(gene_ids=[], data_points=[datapoint])
        prediction_data = PredictionData(adapter)
        dataset = CITPredictionDataset(prediction_data)
        
        # Get item
        image_tensor, sample_info = dataset[0]
        
        # Check image tensor
        assert isinstance(image_tensor, torch.Tensor)
        assert image_tensor.shape == (3, 224, 224)  # [C, H, W]
        assert image_tensor.dtype == torch.float32
        
        # Check sample info
        assert isinstance(sample_info, dict)
        assert sample_info['image_name'] == "test_prediction_image.png"
        assert sample_info['x'] == 150
        assert sample_info['y'] == 250
        assert sample_info['barcode'] == "ABC123"
        assert sample_info['wsi_id'] == "WSI001"
    
    def test_image_transformations(self, temp_dir):
        """Test that image transformations are applied correctly."""
        # Create test image with specific size
        original_img = Image.fromarray(np.random.randint(0, 255, (512, 256, 3), dtype=np.uint8))
        img_path = os.path.join(temp_dir, "large_prediction_image.png")
        original_img.save(img_path)
        
        # Create prediction data point
        datapoint = PredictionDataPoint(
            x=100,
            y=200,
            img_patch_path=img_path,
            wsi_id="WSI001",
            barcode="ABC123"
        )
        
        # Create dataset
        adapter = MockCITPredictionDataAdapter(gene_ids=[], data_points=[datapoint])
        prediction_data = PredictionData(adapter)
        dataset = CITPredictionDataset(prediction_data)
        
        # Get transformed image
        image_tensor, _ = dataset[0]
        
        # Check that image was resized to 224x224
        assert image_tensor.shape == (3, 224, 224)
        
        # Check that normalization was applied (ImageNet stats)
        # Normalized values should be roughly in range [-2, 2] for ImageNet normalization
        assert image_tensor.min() >= -3.0  # Allow some margin
        assert image_tensor.max() <= 3.0
    
    def test_len_method(self, temp_dir):
        """Test __len__ method returns correct length."""
        # Create multiple prediction data points
        data_points = []
        for i in range(3):
            img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
            img_path = os.path.join(temp_dir, f"prediction_image_{i}.png")
            img.save(img_path)
            
            datapoint = PredictionDataPoint(
                x=100 + i,
                y=200 + i,
                img_patch_path=img_path,
                wsi_id=f"WSI{i:03d}",
                barcode=f"ABC{i:03d}"
            )
            data_points.append(datapoint)
        
        # Create dataset
        adapter = MockCITPredictionDataAdapter(gene_ids=[], data_points=data_points)
        prediction_data = PredictionData(adapter)
        dataset = CITPredictionDataset(prediction_data)
        
        assert len(dataset) == 3
    
    def test_sample_info_completeness(self, temp_dir):
        """Test that sample_info contains all required fields."""
        # Create test image with specific name
        img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        img_path = os.path.join(temp_dir, "specific_name_image.png")
        img.save(img_path)
        
        # Create prediction data point
        datapoint = PredictionDataPoint(
            x=42,
            y=84,
            img_patch_path=img_path,
            wsi_id="SPECIAL_WSI",
            barcode="SPECIAL_BARCODE"
        )
        
        adapter = MockCITPredictionDataAdapter(gene_ids=[], data_points=[datapoint])
        prediction_data = PredictionData(adapter)
        dataset = CITPredictionDataset(prediction_data)
        
        _, sample_info = dataset[0]
        
        # Check all required fields exist
        required_fields = ['image_name', 'x', 'y', 'barcode', 'wsi_id']
        for field in required_fields:
            assert field in sample_info
        
        # Check field values
        assert sample_info['image_name'] == "specific_name_image.png"
        assert sample_info['x'] == 42
        assert sample_info['y'] == 84
        assert sample_info['barcode'] == "SPECIAL_BARCODE"
        assert sample_info['wsi_id'] == "SPECIAL_WSI"
    
    def test_sample_info_with_none_values(self, temp_dir):
        """Test sample_info when barcode or wsi_id are None."""
        # Create test image
        img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        img_path = os.path.join(temp_dir, "test_image.png")
        img.save(img_path)
        
        # Create prediction data point with None values
        datapoint = PredictionDataPoint(
            x=100,
            y=200,
            img_patch_path=img_path,
            wsi_id=None,  # None value
            barcode=None  # None value
        )
        
        adapter = MockCITPredictionDataAdapter(gene_ids=[], data_points=[datapoint])
        prediction_data = PredictionData(adapter)
        dataset = CITPredictionDataset(prediction_data)
        
        _, sample_info = dataset[0]
        
        # Check that None values are preserved
        assert sample_info['barcode'] is None
        assert sample_info['wsi_id'] is None
        assert sample_info['x'] == 100
        assert sample_info['y'] == 200
        assert sample_info['image_name'] == "test_image.png"
    
    def test_index_out_of_bounds(self, temp_dir):
        """Test accessing index out of bounds."""
        # Create single prediction data point
        img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        img_path = os.path.join(temp_dir, "test_image.png")
        img.save(img_path)
        
        datapoint = PredictionDataPoint(
            x=100,
            y=200,
            img_patch_path=img_path,
            wsi_id="WSI001",
            barcode="ABC123"
        )
        
        adapter = MockCITPredictionDataAdapter(gene_ids=[], data_points=[datapoint])
        prediction_data = PredictionData(adapter)
        dataset = CITPredictionDataset(prediction_data)
        
        # Valid index
        assert dataset[0] is not None
        
        # Invalid indices
        with pytest.raises(IndexError):
            dataset[1]
        
        with pytest.raises(IndexError):
            dataset[-2]
    
    def test_missing_image_file(self, temp_dir):
        """Test dataset behavior when image file is missing."""
        # For this test, we need to bypass the PredictionData validation since the error
        # should occur at the Dataset level when trying to load the image
        
        # Create a valid image first to pass PredictionData validation
        valid_img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        valid_path = os.path.join(temp_dir, "valid.png")
        valid_img.save(valid_path)
        
        datapoint = PredictionDataPoint(
            x=100,
            y=200,
            img_patch_path=valid_path,  # Use valid path for PredictionData validation
            wsi_id="WSI001",
            barcode="ABC123"
        )
        
        adapter = MockCITPredictionDataAdapter(gene_ids=[], data_points=[datapoint])
        prediction_data = PredictionData(adapter)
        dataset = CITPredictionDataset(prediction_data)
        
        # Now modify the path after PredictionData validation to simulate missing file
        dataset.prediction_data.adapter.data_points[0].img_patch_path = os.path.join(temp_dir, "nonexistent.png")
        
        # Should raise error when trying to load missing image
        with pytest.raises(FileNotFoundError):
            dataset[0]
    
    def test_invalid_image_format(self, temp_dir):
        """Test dataset behavior with invalid image format."""
        # Create a text file instead of image
        text_path = os.path.join(temp_dir, "not_an_image.txt")
        with open(text_path, 'w') as f:
            f.write("This is not an image")
        
        datapoint = PredictionDataPoint(
            x=100,
            y=200,
            img_patch_path=text_path,
            wsi_id="WSI001",
            barcode="ABC123"
        )
        
        adapter = MockCITPredictionDataAdapter(gene_ids=[], data_points=[datapoint])
        prediction_data = PredictionData(adapter)
        dataset = CITPredictionDataset(prediction_data)
        
        # Should raise error when trying to open invalid image
        with pytest.raises((OSError, Image.UnidentifiedImageError)):
            dataset[0]
    
    def test_coordinate_data_types(self, temp_dir):
        """Test that coordinates can be different numeric types."""
        # Create test image
        img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        img_path = os.path.join(temp_dir, "test_image.png")
        img.save(img_path)
        
        # Create prediction data point with float coordinates
        datapoint = PredictionDataPoint(
            x=100.5,  # float
            y=200.7,  # float
            img_patch_path=img_path,
            wsi_id="WSI001",
            barcode="ABC123"
        )
        
        adapter = MockCITPredictionDataAdapter(gene_ids=[], data_points=[datapoint])
        prediction_data = PredictionData(adapter)
        dataset = CITPredictionDataset(prediction_data)
        
        _, sample_info = dataset[0]
        
        # Check that coordinate values are preserved
        assert sample_info['x'] == 100.5
        assert sample_info['y'] == 200.7
    
    def test_image_name_extraction(self, temp_dir):
        """Test correct extraction of image name from path."""
        # Create test image in subdirectory
        subdir = os.path.join(temp_dir, "subdir")
        os.makedirs(subdir, exist_ok=True)
        
        img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        img_path = os.path.join(subdir, "complex_image_name.png")
        img.save(img_path)
        
        datapoint = PredictionDataPoint(
            x=100,
            y=200,
            img_patch_path=img_path,
            wsi_id="WSI001",
            barcode="ABC123"
        )
        
        adapter = MockCITPredictionDataAdapter(gene_ids=[], data_points=[datapoint])
        prediction_data = PredictionData(adapter)
        dataset = CITPredictionDataset(prediction_data)
        
        _, sample_info = dataset[0]
        
        # Should extract only the filename, not the full path
        assert sample_info['image_name'] == "complex_image_name.png"
    
    def test_no_gene_expression_data(self, temp_dir):
        """Test that prediction dataset doesn't return gene expression data."""
        # Create test image
        img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        img_path = os.path.join(temp_dir, "test_image.png")
        img.save(img_path)
        
        datapoint = PredictionDataPoint(
            x=100,
            y=200,
            img_patch_path=img_path,
            wsi_id="WSI001",
            barcode="ABC123"
        )
        
        adapter = MockCITPredictionDataAdapter(gene_ids=[], data_points=[datapoint])
        prediction_data = PredictionData(adapter)
        dataset = CITPredictionDataset(prediction_data)
        
        result = dataset[0]
        
        # Should return only (image_tensor, sample_info), no gene expression
        assert len(result) == 2
        image_tensor, sample_info = result
        
        assert isinstance(image_tensor, torch.Tensor)
        assert isinstance(sample_info, dict)
        
        # Sample info should not contain gene expression data
        assert 'gene_expression' not in sample_info
    
    def test_dtype_consistency(self, temp_dir):
        """Test that tensor dtypes are consistent."""
        # Create test image
        img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        img_path = os.path.join(temp_dir, "test_image.png")
        img.save(img_path)
        
        datapoint = PredictionDataPoint(
            x=100,
            y=200,
            img_patch_path=img_path,
            wsi_id="WSI001",
            barcode="ABC123"
        )
        
        adapter = MockCITPredictionDataAdapter(gene_ids=[], data_points=[datapoint])
        prediction_data = PredictionData(adapter)
        dataset = CITPredictionDataset(prediction_data)
        
        image_tensor, _ = dataset[0]
        
        # Image tensor should be float32
        assert image_tensor.dtype == torch.float32
    
    @pytest.mark.parametrize("img_size", [(100, 100), (300, 400), (500, 500), (800, 600)])
    def test_different_input_image_sizes(self, temp_dir, img_size):
        """Test dataset with different input image sizes."""
        # Create test image with specific size
        img = Image.fromarray(np.random.randint(0, 255, (*img_size, 3), dtype=np.uint8))
        img_path = os.path.join(temp_dir, f"image_{img_size[0]}x{img_size[1]}.png")
        img.save(img_path)
        
        datapoint = PredictionDataPoint(
            x=100,
            y=200,
            img_patch_path=img_path,
            wsi_id="WSI001",
            barcode="ABC123"
        )
        
        adapter = MockCITPredictionDataAdapter(gene_ids=[], data_points=[datapoint])
        prediction_data = PredictionData(adapter)
        dataset = CITPredictionDataset(prediction_data)
        
        image_tensor, _ = dataset[0]
        
        # All images should be resized to 224x224 regardless of input size
        assert image_tensor.shape == (3, 224, 224)
