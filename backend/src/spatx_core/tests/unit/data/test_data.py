"""Tests for Data and PredictionData classes."""
import pytest
import tempfile
import os
from PIL import Image
import numpy as np

from spatx_core.data.data import Data, PredictionData
from spatx_core.data.data_point import DataPoint, PredictionDataPoint
from spatx_core.data_adapters.base_data_adapter import BaseDataAdapter


class MockDataAdapter(BaseDataAdapter):
    """Mock data adapter for testing."""
    
    name = "MockDataAdapter"
    
    def __init__(self, gene_ids, data_points=None, empty=False):
        self.gene_ids = gene_ids
        self.empty = empty
        if data_points:
            self.data_points = data_points
        else:
            self.data_points = []
    
    def __getitem__(self, idx):
        if self.empty:
            raise IndexError("Empty adapter")
        return self.data_points[idx]
    
    def __len__(self):
        return 0 if self.empty else len(self.data_points)


class MockPredictionDataAdapter(BaseDataAdapter):
    """Mock prediction data adapter for testing."""
    
    name = "MockPredictionDataAdapter"
    
    def __init__(self, gene_ids, data_points=None, empty=False):
        self.gene_ids = gene_ids
        self.empty = empty
        if data_points:
            self.data_points = data_points
        else:
            self.data_points = []
    
    def __getitem__(self, idx):
        if self.empty:
            raise IndexError("Empty adapter")
        return self.data_points[idx]
    
    def __len__(self):
        return 0 if self.empty else len(self.data_points)


class MockInvalidAdapter(BaseDataAdapter):
    """Mock adapter that returns invalid data types."""
    
    name = "MockInvalidAdapter"
    
    def __init__(self, gene_ids):
        self.gene_ids = gene_ids
    
    def __getitem__(self, idx):
        # Returns invalid data type
        return {"invalid": "data"}
    
    def __len__(self):
        return 1


class TestData:
    """Test cases for Data class."""
    
    def test_data_initialization_success(self, temp_dir):
        """Test successful Data initialization with valid adapter."""
        # Create test image
        img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        img_path = os.path.join(temp_dir, "test_image.png")
        img.save(img_path)
        
        # Create valid data points
        gene_expression = {"GENE1": 1.5, "GENE2": 2.3}
        datapoint = DataPoint(
            x=100,
            y=200,
            img_patch_path=img_path,
            gene_expression=gene_expression,
            wsi_id="WSI001",
            barcode="ABC123"
        )
        
        # Create mock adapter
        adapter = MockDataAdapter(
            gene_ids=["GENE1", "GENE2"],
            data_points=[datapoint]
        )
        
        # Initialize Data
        data = Data(adapter)
        
        assert len(data) == 1
        assert data[0] == datapoint
        assert data.adapter == adapter
    
    def test_data_initialization_empty_adapter(self):
        """Test Data initialization with empty adapter."""
        adapter = MockDataAdapter(gene_ids=["GENE1"], empty=True)
        
        with pytest.raises(RuntimeError, match="providing zero data length"):
            Data(adapter)
    
    def test_data_initialization_invalid_adapter(self, temp_dir):
        """Test Data initialization with adapter returning invalid data type."""
        adapter = MockInvalidAdapter(gene_ids=["GENE1"])
        
        with pytest.raises(ValueError, match="not returning a standard data point"):
            Data(adapter)
    
    def test_data_getitem_and_len(self, temp_dir):
        """Test Data __getitem__ and __len__ methods."""
        # Create multiple test images and data points
        data_points = []
        for i in range(3):
            img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
            img_path = os.path.join(temp_dir, f"test_image_{i}.png")
            img.save(img_path)
            
            gene_expression = {"GENE1": float(i), "GENE2": float(i + 1)}
            datapoint = DataPoint(
                x=100 + i,
                y=200 + i,
                img_patch_path=img_path,
                gene_expression=gene_expression,
                wsi_id=f"WSI{i:03d}",
                barcode=f"ABC{i:03d}"
            )
            data_points.append(datapoint)
        
        adapter = MockDataAdapter(
            gene_ids=["GENE1", "GENE2"],
            data_points=data_points
        )
        
        data = Data(adapter)
        
        assert len(data) == 3
        for i in range(3):
            assert data[i] == data_points[i]
            assert data[i].x == 100 + i
            assert data[i].y == 200 + i
    
    def test_data_validation_with_invalid_datapoint(self, temp_dir):
        """Test Data validation when adapter returns invalid DataPoint."""
        # Create a DataPoint with missing image file
        nonexistent_path = os.path.join(temp_dir, "nonexistent.png")
        gene_expression = {"GENE1": 1.5}
        
        invalid_datapoint = DataPoint(
            x=100,
            y=200,
            img_patch_path=nonexistent_path,  # File doesn't exist
            gene_expression=gene_expression,
            wsi_id="WSI001",
            barcode="ABC123"
        )
        
        adapter = MockDataAdapter(
            gene_ids=["GENE1"],
            data_points=[invalid_datapoint]
        )
        
        # The error comes directly from validate_datapoint, not from Data wrapper
        with pytest.raises(ValueError, match="There is no image at path provided by adapter"):
            Data(adapter)


class TestPredictionData:
    """Test cases for PredictionData class."""
    
    def test_prediction_data_initialization_success(self, temp_dir):
        """Test successful PredictionData initialization with valid adapter."""
        # Create test image
        img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        img_path = os.path.join(temp_dir, "test_image.png")
        img.save(img_path)
        
        # Create valid prediction data point
        datapoint = PredictionDataPoint(
            x=100,
            y=200,
            img_patch_path=img_path,
            wsi_id="WSI001",
            barcode="ABC123"
        )
        
        # Create mock adapter
        adapter = MockPredictionDataAdapter(
            gene_ids=[],  # No genes needed for prediction data
            data_points=[datapoint]
        )
        
        # Initialize PredictionData
        data = PredictionData(adapter)
        
        assert len(data) == 1
        assert data[0] == datapoint
        assert data.adapter == adapter
    
    def test_prediction_data_initialization_empty_adapter(self):
        """Test PredictionData initialization with empty adapter."""
        adapter = MockPredictionDataAdapter(gene_ids=[], empty=True)
        
        with pytest.raises(RuntimeError, match="providing zero data length"):
            PredictionData(adapter)
    
    def test_prediction_data_initialization_invalid_adapter(self):
        """Test PredictionData initialization with adapter returning invalid data type."""
        adapter = MockInvalidAdapter(gene_ids=[])
        
        with pytest.raises(ValueError, match="not returning a standard prediction data point"):
            PredictionData(adapter)
    
    def test_prediction_data_getitem_and_len(self, temp_dir):
        """Test PredictionData __getitem__ and __len__ methods."""
        # Create multiple test images and prediction data points
        data_points = []
        for i in range(3):
            img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
            img_path = os.path.join(temp_dir, f"test_image_{i}.png")
            img.save(img_path)
            
            datapoint = PredictionDataPoint(
                x=100 + i,
                y=200 + i,
                img_patch_path=img_path,
                wsi_id=f"WSI{i:03d}",
                barcode=f"ABC{i:03d}"
            )
            data_points.append(datapoint)
        
        adapter = MockPredictionDataAdapter(
            gene_ids=[],
            data_points=data_points
        )
        
        data = PredictionData(adapter)
        
        assert len(data) == 3
        for i in range(3):
            assert data[i] == data_points[i]
            assert data[i].x == 100 + i
            assert data[i].y == 200 + i
    
    def test_prediction_data_validation_with_invalid_datapoint(self, temp_dir):
        """Test PredictionData validation when adapter returns invalid PredictionDataPoint."""
        # Create a PredictionDataPoint with missing image file
        nonexistent_path = os.path.join(temp_dir, "nonexistent.png")
        
        invalid_datapoint = PredictionDataPoint(
            x=100,
            y=200,
            img_patch_path=nonexistent_path,  # File doesn't exist
            wsi_id="WSI001",
            barcode="ABC123"
        )
        
        adapter = MockPredictionDataAdapter(
            gene_ids=[],
            data_points=[invalid_datapoint]
        )
        
        # The error comes directly from validate_datapoint, not from PredictionData wrapper
        with pytest.raises(ValueError, match="There is no image at path provided by adapter"):
            PredictionData(adapter)
