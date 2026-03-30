"""Tests for BreastPredictionDataAdapter class."""
import pytest
import os
import pandas as pd
import numpy as np
from PIL import Image

from spatx_core.data_adapters.breast_csv import BreastPredictionDataAdapter
from spatx_core.data.data_point import PredictionDataPoint


class TestBreastPredictionDataAdapter:
    """Test cases for BreastPredictionDataAdapter class."""
    
    def test_initialization_success(self, create_test_image, create_prediction_data):
        """Test successful BreastPredictionDataAdapter initialization."""
        # Create test data
        csv_path = create_prediction_data(num_samples=5)
        
        # Create test images for the data
        for i in range(5):
            barcode = f"{str(i).zfill(3)}x{str(i).zfill(3)}"
            image_path = create_test_image(barcode=barcode, wsi_id="NCBI785")
        
        image_dir = str(image_path).rsplit("/", 1)[0]
        
        # Initialize adapter
        adapter = BreastPredictionDataAdapter(
            image_dir=image_dir,
            prediction_csv=csv_path,
            wsi_ids=["NCBI785"]
        )
        
        assert adapter is not None
        assert len(adapter) > 0
        assert adapter.name == "Breast Prediction Data Adapter"
        assert adapter.image_dir == image_dir
        assert adapter.prediction_csv == csv_path
        assert adapter.wsi_ids == ["NCBI785"]
    
    def test_initialization_invalid_image_dir(self, create_prediction_data):
        """Test BreastPredictionDataAdapter initialization with invalid image directory."""
        csv_path = create_prediction_data()
        
        with pytest.raises(ValueError, match="The path .* does not exist"):
            BreastPredictionDataAdapter(
                image_dir="/nonexistent/path",
                prediction_csv=csv_path,
                wsi_ids=["NCBI785"]
            )
    
    def test_initialization_invalid_csv_file(self, temp_dir):
        """Test BreastPredictionDataAdapter initialization with invalid CSV file."""
        with pytest.raises(FileNotFoundError, match="The file .* does not exist"):
            BreastPredictionDataAdapter(
                image_dir=temp_dir,
                prediction_csv="/nonexistent/file.csv",
                wsi_ids=["NCBI785"]
            )
    
    def test_initialization_missing_required_columns(self, create_test_image, temp_dir):
        """Test BreastPredictionDataAdapter initialization with missing required columns."""
        # Create CSV missing required columns
        data = {
            'barcode': ['000x000', '001x001'],
            'id': ['NCBI785', 'NCBI785'],
            # Missing 'x_pixel' and 'y_pixel'
            'combined_text': ['Sample text'] * 2
        }
        df = pd.DataFrame(data)
        csv_path = os.path.join(temp_dir, "incomplete_prediction.csv")
        df.to_csv(csv_path, index=False)
        
        image_path = create_test_image()
        image_dir = str(image_path).rsplit("/", 1)[0]
        
        with pytest.raises(ValueError, match="The following required columns are missing"):
            BreastPredictionDataAdapter(
                image_dir=image_dir,
                prediction_csv=csv_path,
                wsi_ids=["NCBI785"]
            )
    
    def test_initialization_missing_wsi_ids(self, create_test_image, create_prediction_data):
        """Test BreastPredictionDataAdapter initialization with missing WSI IDs in CSV."""
        csv_path = create_prediction_data()
        image_path = create_test_image()
        image_dir = str(image_path).rsplit("/", 1)[0]
        
        with pytest.raises(ValueError, match="The following WSI IDs are missing in the CSV"):
            BreastPredictionDataAdapter(
                image_dir=image_dir,
                prediction_csv=csv_path,
                wsi_ids=["NONEXISTENT_WSI"]
            )
    
    def test_getitem_success(self, create_test_image, create_prediction_data):
        """Test successful data retrieval via __getitem__."""
        # Create test data
        csv_path = create_prediction_data(num_samples=3)
        barcode = "000x000"  # First item in generated CSV
        image_path = create_test_image(barcode=barcode, wsi_id="NCBI785")
        image_dir = str(image_path).rsplit("/", 1)[0]
        
        # Initialize adapter
        adapter = BreastPredictionDataAdapter(
            image_dir=image_dir,
            prediction_csv=csv_path,
            wsi_ids=["NCBI785"]
        )
        
        # Get first item and test PredictionDataPoint structure
        data_point = adapter[0]
        
        assert isinstance(data_point, PredictionDataPoint)
        assert data_point.wsi_id == "NCBI785"
        assert data_point.barcode == barcode
        assert isinstance(data_point.x, (int, float))
        assert isinstance(data_point.y, (int, float))
        assert data_point.img_patch_path.endswith(f"{barcode}_NCBI785.png")
        assert os.path.exists(data_point.img_patch_path)
        
        # Test data point validation
        assert data_point.validate_datapoint("BreastPredictionDataAdapter") is True
    
    def test_getitem_missing_image(self, create_prediction_data, temp_dir):
        """Test __getitem__ when corresponding image file is missing."""
        csv_path = create_prediction_data()
        
        # Initialize adapter without creating corresponding image files
        adapter = BreastPredictionDataAdapter(
            image_dir=temp_dir,
            prediction_csv=csv_path,
            wsi_ids=["NCBI785"]
        )
        
        # Should raise FileNotFoundError when trying to access data
        with pytest.raises(FileNotFoundError, match="Image patch not found"):
            adapter[0]
    
    def test_len_method(self, create_test_image, create_prediction_data):
        """Test __len__ method returns correct length."""
        num_samples = 7
        
        # Create test data
        csv_path = create_prediction_data(num_samples=num_samples)
        
        # Create images for all samples
        for i in range(num_samples):
            barcode = f"{str(i).zfill(3)}x{str(i).zfill(3)}"
            image_path = create_test_image(barcode=barcode, wsi_id="NCBI785")
        
        image_dir = str(image_path).rsplit("/", 1)[0]
        
        adapter = BreastPredictionDataAdapter(
            image_dir=image_dir,
            prediction_csv=csv_path,
            wsi_ids=["NCBI785"]
        )
        
        assert len(adapter) == num_samples
    
    def test_filtering_by_wsi_ids(self, create_test_image, temp_dir):
        """Test that adapter correctly filters data by WSI IDs."""
        # Create CSV with multiple WSI IDs
        barcodes = [f"{str(i).zfill(3)}x{str(i).zfill(3)}" for i in range(6)]
        
        data = {
            'barcode': barcodes,
            'id': ["NCBI785", "NCBI785", "NCBI786", "NCBI786", "NCBI787", "NCBI787"],
            'x_pixel': np.random.uniform(0, 40000, 6),
            'y_pixel': np.random.uniform(0, 80000, 6),
            'combined_text': ['Sample text'] * 6
        }
        df = pd.DataFrame(data)
        csv_path = os.path.join(temp_dir, "multi_wsi_prediction.csv")
        df.to_csv(csv_path, index=False)
        
        # Create images for all barcodes
        image_dir = os.path.join(temp_dir, "images")
        os.makedirs(image_dir, exist_ok=True)
        for i, wsi_id in enumerate(["NCBI785", "NCBI785", "NCBI786", "NCBI786", "NCBI787", "NCBI787"]):
            barcode = barcodes[i]
            img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
            img_path = os.path.join(image_dir, f"{barcode}_{wsi_id}.png")
            img.save(img_path)
        
        # Test filtering to only NCBI785
        adapter = BreastPredictionDataAdapter(
            image_dir=image_dir,
            prediction_csv=csv_path,
            wsi_ids=["NCBI785"]
        )
        
        assert len(adapter) == 2  # Only 2 samples with NCBI785
        for i in range(len(adapter)):
            data_point = adapter[i]
            assert data_point.wsi_id == "NCBI785"
        
        # Test filtering to multiple WSIs
        adapter_multi = BreastPredictionDataAdapter(
            image_dir=image_dir,
            prediction_csv=csv_path,
            wsi_ids=["NCBI785", "NCBI786"]
        )
        
        assert len(adapter_multi) == 4  # 2 from each WSI
        wsi_ids_found = [adapter_multi[i].wsi_id for i in range(len(adapter_multi))]
        assert all(wsi_id in ["NCBI785", "NCBI786"] for wsi_id in wsi_ids_found)
    
    def test_index_out_of_bounds(self, create_test_image, create_prediction_data):
        """Test accessing index out of bounds."""
        num_samples = 3
        
        csv_path = create_prediction_data(num_samples=num_samples)
        
        for i in range(num_samples):
            barcode = f"{str(i).zfill(3)}x{str(i).zfill(3)}"
            image_path = create_test_image(barcode=barcode, wsi_id="NCBI785")
        
        image_dir = str(image_path).rsplit("/", 1)[0]
        
        adapter = BreastPredictionDataAdapter(
            image_dir=image_dir,
            prediction_csv=csv_path,
            wsi_ids=["NCBI785"]
        )
        
        # Valid indices
        for i in range(num_samples):
            assert adapter[i] is not None
        
        # Invalid indices
        with pytest.raises(IndexError):
            adapter[num_samples]  # One past the end
        
        with pytest.raises(IndexError):
            adapter[-num_samples - 1]  # Too negative
    
    def test_coordinate_types(self, create_test_image, create_prediction_data):
        """Test that x and y coordinates have correct types."""
        csv_path = create_prediction_data(num_samples=1)
        barcode = "000x000"
        image_path = create_test_image(barcode=barcode, wsi_id="NCBI785")
        image_dir = str(image_path).rsplit("/", 1)[0]
        
        adapter = BreastPredictionDataAdapter(
            image_dir=image_dir,
            prediction_csv=csv_path,
            wsi_ids=["NCBI785"]
        )
        
        data_point = adapter[0]
        
        # Coordinates should be numeric
        assert isinstance(data_point.x, (int, float))
        assert isinstance(data_point.y, (int, float))
        assert not pd.isna(data_point.x)
        assert not pd.isna(data_point.y)
    
    def test_no_gene_expression_data(self, create_test_image, create_prediction_data):
        """Test that PredictionDataPoint doesn't have gene expression data."""
        csv_path = create_prediction_data(num_samples=1)
        barcode = "000x000"
        image_path = create_test_image(barcode=barcode, wsi_id="NCBI785")
        image_dir = str(image_path).rsplit("/", 1)[0]
        
        adapter = BreastPredictionDataAdapter(
            image_dir=image_dir,
            prediction_csv=csv_path,
            wsi_ids=["NCBI785"]
        )
        
        data_point = adapter[0]
        
        # PredictionDataPoint should not have gene_expression attribute
        assert not hasattr(data_point, 'gene_expression')
        assert isinstance(data_point, PredictionDataPoint)
    
    def test_adapter_with_extra_columns(self, create_test_image, temp_dir):
        """Test adapter works when CSV has extra columns beyond required ones."""
        # Create CSV with extra columns
        barcodes = ['000x000', '001x001']
        data = {
            'barcode': barcodes,
            'id': ['NCBI785', 'NCBI785'],
            'x_pixel': [1000.0, 2000.0],
            'y_pixel': [1500.0, 2500.0],
            'combined_text': ['Sample text'] * 2,
            'extra_column1': ['extra1', 'extra2'],
            'extra_column2': [10, 20],
            'GENE1': [1.5, 2.5],  # Gene column that should be ignored
        }
        df = pd.DataFrame(data)
        csv_path = os.path.join(temp_dir, "extra_cols_prediction.csv")
        df.to_csv(csv_path, index=False)
        
        # Create images
        image_dir = os.path.join(temp_dir, "images")
        os.makedirs(image_dir, exist_ok=True)
        for i, barcode in enumerate(barcodes):
            img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
            img_path = os.path.join(image_dir, f"{barcode}_NCBI785.png")
            img.save(img_path)
        
        adapter = BreastPredictionDataAdapter(
            image_dir=image_dir,
            prediction_csv=csv_path,
            wsi_ids=["NCBI785"]
        )
        
        assert len(adapter) == 2
        
        # Verify data points are created correctly
        for i in range(len(adapter)):
            data_point = adapter[i]
            assert isinstance(data_point, PredictionDataPoint)
            assert data_point.wsi_id == "NCBI785"
            assert data_point.barcode in barcodes
            # Should not have gene expression even though GENE1 column exists
            assert not hasattr(data_point, 'gene_expression')
