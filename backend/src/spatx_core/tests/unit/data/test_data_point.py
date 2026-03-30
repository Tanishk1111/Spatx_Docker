"""Tests for DataPoint and PredictionDataPoint classes."""
import pytest
import tempfile
import os
import warnings
from PIL import Image
import numpy as np

from spatx_core.data.data_point import DataPoint, PredictionDataPoint


class TestDataPoint:
    """Test cases for DataPoint class."""
    
    def test_datapoint_initialization(self, temp_dir):
        """Test basic DataPoint initialization."""
        # Create a test image
        img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        img_path = os.path.join(temp_dir, "test_image.png")
        img.save(img_path)
        
        gene_expression = {"GENE1": 1.5, "GENE2": 2.3, "GENE3": 0.8}
        
        datapoint = DataPoint(
            x=100,
            y=200,
            img_patch_path=img_path,
            gene_expression=gene_expression,
            wsi_id="WSI001",
            barcode="ABC123"
        )
        
        assert datapoint.x == 100
        assert datapoint.y == 200
        assert datapoint.img_patch_path == img_path
        assert datapoint.gene_expression == gene_expression
        assert datapoint.wsi_id == "WSI001"
        assert datapoint.barcode == "ABC123"
    
    def test_datapoint_validation_success(self, temp_dir):
        """Test successful DataPoint validation."""
        # Create a test image
        img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        img_path = os.path.join(temp_dir, "test_image.png")
        img.save(img_path)
        
        gene_expression = {"GENE1": 1.5, "GENE2": 2.3}
        
        datapoint = DataPoint(
            x=100.0,
            y=200.0,
            img_patch_path=img_path,
            gene_expression=gene_expression,
            wsi_id="WSI001",
            barcode="ABC123"
        )
        
        assert datapoint.validate_datapoint("TestAdapter") is True
    
    def test_datapoint_validation_missing_image(self, temp_dir):
        """Test DataPoint validation with missing image file."""
        nonexistent_path = os.path.join(temp_dir, "nonexistent.png")
        gene_expression = {"GENE1": 1.5}
        
        datapoint = DataPoint(
            x=100,
            y=200,
            img_patch_path=nonexistent_path,
            gene_expression=gene_expression,
            wsi_id="WSI001",
            barcode="ABC123"
        )
        
        with pytest.raises(ValueError, match="There is no image at path"):
            datapoint.validate_datapoint("TestAdapter")
    
    def test_datapoint_validation_invalid_gene_expression(self, temp_dir):
        """Test DataPoint validation with invalid gene expression format."""
        # Create a test image
        img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        img_path = os.path.join(temp_dir, "test_image.png")
        img.save(img_path)
        
        # Invalid gene expression (not a dict)
        invalid_gene_expression = ["GENE1", "GENE2"]
        
        datapoint = DataPoint(
            x=100,
            y=200,
            img_patch_path=img_path,
            gene_expression=invalid_gene_expression,  # type: ignore
            wsi_id="WSI001",
            barcode="ABC123"
        )
        
        with pytest.raises(ValueError, match="not returning a gene_expression object in format dict"):
            datapoint.validate_datapoint("TestAdapter")
    
    def test_datapoint_validation_warnings(self, temp_dir):
        """Test DataPoint validation with warning conditions."""
        # Create a test image
        img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        img_path = os.path.join(temp_dir, "test_image.png")
        img.save(img_path)
        
        gene_expression = {"GENE1": 1.5}
        
        # Test missing wsi_id warning
        datapoint = DataPoint(
            x=100,
            y=200,
            img_patch_path=img_path,
            gene_expression=gene_expression,
            wsi_id=None,  # Will trigger warning
            barcode="ABC123"
        )
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            datapoint.validate_datapoint("TestAdapter")
            assert len(w) >= 1
            assert "without wsi id" in str(w[0].message)
        
        # Test missing barcode warning
        datapoint = DataPoint(
            x=100,
            y=200,
            img_patch_path=img_path,
            gene_expression=gene_expression,
            wsi_id="WSI001",
            barcode=None  # Will trigger warning
        )
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            datapoint.validate_datapoint("TestAdapter")
            assert len(w) >= 1
            assert "without barcode" in str(w[0].message)
        
        # Test invalid coordinate types warning
        datapoint = DataPoint(
            x="100",  # String instead of number - will trigger warning
            y="200",  # String instead of number - will trigger warning
            img_patch_path=img_path,
            gene_expression=gene_expression,
            wsi_id="WSI001",
            barcode="ABC123"
        )
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            datapoint.validate_datapoint("TestAdapter")
            assert len(w) >= 1
            assert "not instances of int | float" in str(w[0].message)
    
    @pytest.mark.parametrize("x_val,y_val", [
        (100, 200),
        (100.5, 200.7),
        (0, 0),
        (-50, -100)
    ])
    def test_datapoint_coordinate_types(self, temp_dir, x_val, y_val):
        """Test DataPoint with different coordinate types."""
        # Create a test image
        img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        img_path = os.path.join(temp_dir, "test_image.png")
        img.save(img_path)
        
        gene_expression = {"GENE1": 1.5}
        
        datapoint = DataPoint(
            x=x_val,
            y=y_val,
            img_patch_path=img_path,
            gene_expression=gene_expression,
            wsi_id="WSI001",
            barcode="ABC123"
        )
        
        assert datapoint.validate_datapoint("TestAdapter") is True


class TestPredictionDataPoint:
    """Test cases for PredictionDataPoint class."""
    
    def test_prediction_datapoint_initialization(self, temp_dir):
        """Test basic PredictionDataPoint initialization."""
        # Create a test image
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
        
        assert datapoint.x == 100
        assert datapoint.y == 200
        assert datapoint.img_patch_path == img_path
        assert datapoint.wsi_id == "WSI001"
        assert datapoint.barcode == "ABC123"
    
    def test_prediction_datapoint_validation_success(self, temp_dir):
        """Test successful PredictionDataPoint validation."""
        # Create a test image
        img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        img_path = os.path.join(temp_dir, "test_image.png")
        img.save(img_path)
        
        datapoint = PredictionDataPoint(
            x=100.0,
            y=200.0,
            img_patch_path=img_path,
            wsi_id="WSI001",
            barcode="ABC123"
        )
        
        assert datapoint.validate_datapoint("TestAdapter") is True
    
    def test_prediction_datapoint_validation_missing_image(self, temp_dir):
        """Test PredictionDataPoint validation with missing image file."""
        nonexistent_path = os.path.join(temp_dir, "nonexistent.png")
        
        datapoint = PredictionDataPoint(
            x=100,
            y=200,
            img_patch_path=nonexistent_path,
            wsi_id="WSI001",
            barcode="ABC123"
        )
        
        with pytest.raises(ValueError, match="There is no image at path"):
            datapoint.validate_datapoint("TestAdapter")
    
    def test_prediction_datapoint_validation_warnings(self, temp_dir):
        """Test PredictionDataPoint validation with warning conditions."""
        # Create a test image
        img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        img_path = os.path.join(temp_dir, "test_image.png")
        img.save(img_path)
        
        # Test missing wsi_id warning
        datapoint = PredictionDataPoint(
            x=100,
            y=200,
            img_patch_path=img_path,
            wsi_id=None,  # Will trigger warning
            barcode="ABC123"
        )
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            datapoint.validate_datapoint("TestAdapter")
            assert len(w) >= 1
            assert "without wsi id" in str(w[0].message)
        
        # Test invalid coordinate types warning
        datapoint = PredictionDataPoint(
            x="100",  # String instead of number - will trigger warning
            y="200",  # String instead of number - will trigger warning
            img_patch_path=img_path,
            wsi_id="WSI001",
            barcode="ABC123"
        )
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            datapoint.validate_datapoint("TestAdapter")
            assert len(w) >= 1
            assert "not instances of int | float" in str(w[0].message)
    
    def test_prediction_datapoint_minimal_initialization(self, temp_dir):
        """Test PredictionDataPoint with minimal required parameters."""
        # Create a test image
        img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        img_path = os.path.join(temp_dir, "test_image.png")
        img.save(img_path)
        
        datapoint = PredictionDataPoint(
            x=100,
            y=200,
            img_patch_path=img_path
        )
        
        assert datapoint.x == 100
        assert datapoint.y == 200
        assert datapoint.img_patch_path == img_path
        assert datapoint.wsi_id is None
        assert datapoint.barcode is None
