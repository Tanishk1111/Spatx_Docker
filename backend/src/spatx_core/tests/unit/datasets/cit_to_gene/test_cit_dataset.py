"""Tests for CITDataset class."""
import pytest
import torch
import numpy as np
from PIL import Image
import os

from spatx_core.datasets.cit_to_gene.dataset import CITDataset
from spatx_core.data.data import Data
from spatx_core.data.data_point import DataPoint
from spatx_core.data_adapters.base_data_adapter import BaseDataAdapter


class MockCITDataAdapter(BaseDataAdapter):
    """Mock data adapter for CITDataset testing."""
    
    name = "MockCITDataAdapter"
    
    def __init__(self, gene_ids, data_points=None):
        self.gene_ids = sorted(gene_ids)  # Keep sorted for consistency
        self.data_points = data_points or []
    
    def __getitem__(self, idx):
        return self.data_points[idx]
    
    def __len__(self):
        return len(self.data_points)


class TestCITDataset:
    """Test cases for CITDataset class."""
    
    def test_initialization_success(self, temp_dir):
        """Test successful CITDataset initialization."""
        # Create test image
        img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        img_path = os.path.join(temp_dir, "test_image.png")
        img.save(img_path)
        
        # Create data point
        gene_expression = {"GENE1": 1.5, "GENE2": 2.3}
        datapoint = DataPoint(
            x=100,
            y=200,
            img_patch_path=img_path,
            gene_expression=gene_expression,
            wsi_id="WSI001",
            barcode="ABC123"
        )
        
        # Create adapter and data
        adapter = MockCITDataAdapter(
            gene_ids=["GENE1", "GENE2"],
            data_points=[datapoint]
        )
        data = Data(adapter)
        
        # Initialize dataset
        dataset = CITDataset(data)
        
        assert dataset is not None
        assert len(dataset) == 1
        assert dataset.data == data
        assert dataset.transform is not None
    
    def test_getitem_success(self, temp_dir):
        """Test successful data retrieval via __getitem__."""
        # Create test image
        img = Image.fromarray(np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8))
        img_path = os.path.join(temp_dir, "test_image.png")
        img.save(img_path)
        
        # Create data point with sorted gene expression
        gene_expression = {"GENE2": 2.3, "GENE1": 1.5, "GENE3": 0.8}  # Unsorted order
        datapoint = DataPoint(
            x=100,
            y=200,
            img_patch_path=img_path,
            gene_expression=gene_expression,
            wsi_id="WSI001",
            barcode="ABC123"
        )
        
        # Create adapter and data
        adapter = MockCITDataAdapter(
            gene_ids=["GENE1", "GENE2", "GENE3"],
            data_points=[datapoint]
        )
        data = Data(adapter)
        dataset = CITDataset(data)
        
        # Get item
        image_tensor, expression_tensor, sample_id = dataset[0]
        
        # Check image tensor
        assert isinstance(image_tensor, torch.Tensor)
        assert image_tensor.shape == (3, 224, 224)  # [C, H, W]
        assert image_tensor.dtype == torch.float32
        
        # Check expression tensor (should be sorted by gene name)
        assert isinstance(expression_tensor, torch.Tensor)
        assert expression_tensor.shape == (3,)  # 3 genes
        assert expression_tensor.dtype == torch.float32
        
        # Check values are sorted by gene name: GENE1, GENE2, GENE3
        expected_values = [1.5, 2.3, 0.8]  # GENE1, GENE2, GENE3
        assert torch.allclose(expression_tensor, torch.tensor(expected_values))
        
        # Check sample ID
        assert isinstance(sample_id, str)
        assert sample_id == "ABC123_WSI001"
    
    def test_image_transformations(self, temp_dir):
        """Test that image transformations are applied correctly."""
        # Create test image with specific size
        original_img = Image.fromarray(np.random.randint(0, 255, (512, 256, 3), dtype=np.uint8))
        img_path = os.path.join(temp_dir, "large_image.png")
        original_img.save(img_path)
        
        # Create data point
        gene_expression = {"GENE1": 1.0}
        datapoint = DataPoint(
            x=100,
            y=200,
            img_patch_path=img_path,
            gene_expression=gene_expression,
            wsi_id="WSI001",
            barcode="ABC123"
        )
        
        # Create dataset
        adapter = MockCITDataAdapter(gene_ids=["GENE1"], data_points=[datapoint])
        data = Data(adapter)
        dataset = CITDataset(data)
        
        # Get transformed image
        image_tensor, _, _ = dataset[0]
        
        # Check that image was resized to 224x224
        assert image_tensor.shape == (3, 224, 224)
        
        # Check that normalization was applied (ImageNet stats)
        # Normalized values should be roughly in range [-2, 2] for ImageNet normalization
        assert image_tensor.min() >= -3.0  # Allow some margin
        assert image_tensor.max() <= 3.0
    
    def test_len_method(self, temp_dir):
        """Test __len__ method returns correct length."""
        # Create multiple data points
        data_points = []
        for i in range(5):
            img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
            img_path = os.path.join(temp_dir, f"image_{i}.png")
            img.save(img_path)
            
            gene_expression = {"GENE1": float(i)}
            datapoint = DataPoint(
                x=100 + i,
                y=200 + i,
                img_patch_path=img_path,
                gene_expression=gene_expression,
                wsi_id=f"WSI{i:03d}",
                barcode=f"ABC{i:03d}"
            )
            data_points.append(datapoint)
        
        # Create dataset
        adapter = MockCITDataAdapter(gene_ids=["GENE1"], data_points=data_points)
        data = Data(adapter)
        dataset = CITDataset(data)
        
        assert len(dataset) == 5
    
    def test_consistent_gene_ordering(self, temp_dir):
        """Test that gene expression values are consistently ordered."""
        # Create test image
        img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        img_path = os.path.join(temp_dir, "test_image.png")
        img.save(img_path)
        
        # Create multiple data points with same genes in different orders
        gene_expression_1 = {"GENE_C": 3.0, "GENE_A": 1.0, "GENE_B": 2.0}
        gene_expression_2 = {"GENE_B": 5.0, "GENE_C": 6.0, "GENE_A": 4.0}
        
        data_points = []
        for i, gene_exp in enumerate([gene_expression_1, gene_expression_2]):
            img_copy = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
            img_path_copy = os.path.join(temp_dir, f"test_image_{i}.png")
            img_copy.save(img_path_copy)
            
            datapoint = DataPoint(
                x=100 + i,
                y=200 + i,
                img_patch_path=img_path_copy,
                gene_expression=gene_exp,
                wsi_id=f"WSI{i:03d}",
                barcode=f"ABC{i:03d}"
            )
            data_points.append(datapoint)
        
        # Create dataset
        adapter = MockCITDataAdapter(
            gene_ids=["GENE_A", "GENE_B", "GENE_C"],
            data_points=data_points
        )
        data = Data(adapter)
        dataset = CITDataset(data)
        
        # Get both samples
        _, expr_tensor_1, _ = dataset[0]
        _, expr_tensor_2, _ = dataset[1]
        
        # Both should have genes in same order: GENE_A, GENE_B, GENE_C
        expected_1 = torch.tensor([1.0, 2.0, 3.0])  # GENE_A, GENE_B, GENE_C
        expected_2 = torch.tensor([4.0, 5.0, 6.0])  # GENE_A, GENE_B, GENE_C
        
        assert torch.allclose(expr_tensor_1, expected_1)
        assert torch.allclose(expr_tensor_2, expected_2)
    
    def test_index_out_of_bounds(self, temp_dir):
        """Test accessing index out of bounds."""
        # Create single data point
        img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        img_path = os.path.join(temp_dir, "test_image.png")
        img.save(img_path)
        
        gene_expression = {"GENE1": 1.0}
        datapoint = DataPoint(
            x=100,
            y=200,
            img_patch_path=img_path,
            gene_expression=gene_expression,
            wsi_id="WSI001",
            barcode="ABC123"
        )
        
        adapter = MockCITDataAdapter(gene_ids=["GENE1"], data_points=[datapoint])
        data = Data(adapter)
        dataset = CITDataset(data)
        
        # Valid index
        assert dataset[0] is not None
        
        # Invalid indices
        with pytest.raises(IndexError):
            dataset[1]
        
        with pytest.raises(IndexError):
            dataset[-2]
    
    def test_missing_image_file(self, temp_dir):
        """Test dataset behavior when image file is missing."""
        # For this test, we need to bypass the Data validation since the error
        # should occur at the Dataset level when trying to load the image
        
        # Create a valid image first to pass Data validation
        valid_img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        valid_path = os.path.join(temp_dir, "valid.png")
        valid_img.save(valid_path)
        
        gene_expression = {"GENE1": 1.0}
        datapoint = DataPoint(
            x=100,
            y=200,
            img_patch_path=valid_path,  # Use valid path for Data validation
            gene_expression=gene_expression,
            wsi_id="WSI001",
            barcode="ABC123"
        )
        
        adapter = MockCITDataAdapter(gene_ids=["GENE1"], data_points=[datapoint])
        data = Data(adapter)
        dataset = CITDataset(data)
        
        # Now modify the path after Data validation to simulate missing file
        dataset.data.adapter.data_points[0].img_patch_path = os.path.join(temp_dir, "nonexistent.png")
        
        # Should raise error when trying to load missing image
        with pytest.raises(FileNotFoundError):
            dataset[0]
    
    def test_invalid_image_format(self, temp_dir):
        """Test dataset behavior with invalid image format."""
        # Create a text file instead of image
        text_path = os.path.join(temp_dir, "not_an_image.txt")
        with open(text_path, 'w') as f:
            f.write("This is not an image")
        
        gene_expression = {"GENE1": 1.0}
        datapoint = DataPoint(
            x=100,
            y=200,
            img_patch_path=text_path,
            gene_expression=gene_expression,
            wsi_id="WSI001",
            barcode="ABC123"
        )
        
        adapter = MockCITDataAdapter(gene_ids=["GENE1"], data_points=[datapoint])
        data = Data(adapter)
        dataset = CITDataset(data)
        
        # Should raise error when trying to open invalid image
        with pytest.raises((OSError, Image.UnidentifiedImageError)):
            dataset[0]
    
    def test_empty_gene_expression(self, temp_dir):
        """Test dataset with empty gene expression."""
        # Create test image
        img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        img_path = os.path.join(temp_dir, "test_image.png")
        img.save(img_path)
        
        # Create data point with empty gene expression
        gene_expression = {}
        datapoint = DataPoint(
            x=100,
            y=200,
            img_patch_path=img_path,
            gene_expression=gene_expression,
            wsi_id="WSI001",
            barcode="ABC123"
        )
        
        adapter = MockCITDataAdapter(gene_ids=[], data_points=[datapoint])
        data = Data(adapter)
        dataset = CITDataset(data)
        
        # Should return empty expression tensor
        image_tensor, expression_tensor, sample_id = dataset[0]
        
        assert image_tensor.shape == (3, 224, 224)
        assert expression_tensor.shape == (0,)  # Empty tensor
        assert sample_id == "ABC123_WSI001"
    
    @pytest.mark.parametrize("gene_count", [1, 5, 10, 50])
    def test_different_gene_counts(self, temp_dir, gene_count):
        """Test dataset with different numbers of genes."""
        # Create test image
        img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        img_path = os.path.join(temp_dir, "test_image.png")
        img.save(img_path)
        
        # Create gene expression with specified number of genes
        gene_ids = [f"GENE_{i:03d}" for i in range(gene_count)]
        gene_expression = {gene: float(i) for i, gene in enumerate(gene_ids)}
        
        datapoint = DataPoint(
            x=100,
            y=200,
            img_patch_path=img_path,
            gene_expression=gene_expression,
            wsi_id="WSI001",
            barcode="ABC123"
        )
        
        adapter = MockCITDataAdapter(gene_ids=gene_ids, data_points=[datapoint])
        data = Data(adapter)
        dataset = CITDataset(data)
        
        # Check output shapes
        image_tensor, expression_tensor, sample_id = dataset[0]
        
        assert image_tensor.shape == (3, 224, 224)
        assert expression_tensor.shape == (gene_count,)
        assert len(expression_tensor) == gene_count
    
    def test_dtype_consistency(self, temp_dir):
        """Test that tensor dtypes are consistent."""
        # Create test image
        img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        img_path = os.path.join(temp_dir, "test_image.png")
        img.save(img_path)
        
        # Create data point
        gene_expression = {"GENE1": 1.5, "GENE2": 2}  # Mix of float and int
        datapoint = DataPoint(
            x=100,
            y=200,
            img_patch_path=img_path,
            gene_expression=gene_expression,
            wsi_id="WSI001",
            barcode="ABC123"
        )
        
        adapter = MockCITDataAdapter(gene_ids=["GENE1", "GENE2"], data_points=[datapoint])
        data = Data(adapter)
        dataset = CITDataset(data)
        
        image_tensor, expression_tensor, _ = dataset[0]
        
        # Both tensors should be float32
        assert image_tensor.dtype == torch.float32
        assert expression_tensor.dtype == torch.float32
