"""Tests for BreastDataAdapter class."""
import pytest
import os
import pandas as pd
import numpy as np
from PIL import Image

from spatx_core.data_adapters.breast_csv import BreastDataAdapter
from spatx_core.data.data_point import DataPoint


class TestBreastDataAdapter:
    """Test cases for BreastDataAdapter class."""
    
    def test_initialization_success(self, create_test_image, create_test_expression_data):
        """Test successful BreastDataAdapter initialization."""
        test_genes = ["ABCC11", "ADH1B", "ADIPOQ", "ANKRD30A", "AQP1"]
        
        # Create test data
        csv_path = create_test_expression_data(gene_ids=test_genes)
        
        # Create test images for the data
        for i in range(10):
            barcode = f"{str(i).zfill(3)}x{str(i).zfill(3)}"
            image_path = create_test_image(barcode=barcode, wsi_id="NCBI785")
        
        image_dir = str(image_path).rsplit("/", 1)[0]
        
        # Initialize adapter
        adapter = BreastDataAdapter(
            image_dir=image_dir,
            breast_csv=csv_path,
            wsi_ids=["NCBI785"],
            gene_ids=test_genes
        )
        
        assert adapter is not None
        assert len(adapter) > 0
        assert adapter.gene_ids == sorted(test_genes)
        assert adapter.name == "Breast Data Adapter"
        assert adapter.image_dir == image_dir
        assert adapter.breast_csv == csv_path
        assert adapter.wsi_ids == ["NCBI785"]
    
    def test_initialization_invalid_image_dir(self, create_test_expression_data):
        """Test BreastDataAdapter initialization with invalid image directory."""
        test_genes = ["ABCC11", "ADH1B"]
        csv_path = create_test_expression_data(gene_ids=test_genes)
        
        with pytest.raises(ValueError, match="The path .* does not exists"):
            BreastDataAdapter(
                image_dir="/nonexistent/path",
                breast_csv=csv_path,
                wsi_ids=["NCBI785"],
                gene_ids=test_genes
            )
    
    def test_initialization_invalid_csv_file(self, temp_dir):
        """Test BreastDataAdapter initialization with invalid CSV file."""
        test_genes = ["ABCC11", "ADH1B"]
        
        with pytest.raises(FileNotFoundError, match="The file .* does not exists"):
            BreastDataAdapter(
                image_dir=temp_dir,
                breast_csv="/nonexistent/file.csv",
                wsi_ids=["NCBI785"],
                gene_ids=test_genes
            )
    
    def test_initialization_missing_wsi_ids(self, create_test_image, create_test_expression_data):
        """Test BreastDataAdapter initialization with missing WSI IDs in CSV."""
        test_genes = ["ABCC11", "ADH1B"]
        csv_path = create_test_expression_data(gene_ids=test_genes)
        image_path = create_test_image()
        image_dir = str(image_path).rsplit("/", 1)[0]
        
        with pytest.raises(ValueError, match="The following WSI IDs are missing in the CSV"):
            BreastDataAdapter(
                image_dir=image_dir,
                breast_csv=csv_path,
                wsi_ids=["NONEXISTENT_WSI"],
                gene_ids=test_genes
            )
    
    def test_initialization_duplicate_gene_ids(self, create_test_image, create_test_expression_data):
        """Test BreastDataAdapter initialization with duplicate gene IDs."""
        test_genes = ["ABCC11", "ADH1B", "ABCC11"]  # Duplicate ABCC11
        csv_path = create_test_expression_data(gene_ids=["ABCC11", "ADH1B"])
        image_path = create_test_image()
        image_dir = str(image_path).rsplit("/", 1)[0]
        
        with pytest.raises(ValueError, match="Duplicate gene IDs found"):
            BreastDataAdapter(
                image_dir=image_dir,
                breast_csv=csv_path,
                wsi_ids=["NCBI785"],
                gene_ids=test_genes
            )
    
    def test_initialization_missing_gene_columns(self, create_test_image, create_test_expression_data):
        """Test BreastDataAdapter initialization with missing gene columns in CSV."""
        # Create CSV with limited genes
        available_genes = ["ABCC11", "ADH1B"]
        csv_path = create_test_expression_data(gene_ids=available_genes)
        image_path = create_test_image()
        image_dir = str(image_path).rsplit("/", 1)[0]
        
        # Request genes that don't exist in CSV
        requested_genes = ["ABCC11", "ADH1B", "NONEXISTENT_GENE"]
        
        with pytest.raises(ValueError, match="The following columns are missing in DataFrame"):
            BreastDataAdapter(
                image_dir=image_dir,
                breast_csv=csv_path,
                wsi_ids=["NCBI785"],
                gene_ids=requested_genes
            )
    
    def test_getitem_success(self, create_test_image, create_test_expression_data):
        """Test successful data retrieval via __getitem__."""
        test_genes = ["ABCC11", "ADH1B", "ADIPOQ"]
        
        # Create test data
        csv_path = create_test_expression_data(gene_ids=test_genes)
        barcode = "000x000"  # First item in generated CSV
        image_path = create_test_image(barcode=barcode, wsi_id="NCBI785")
        image_dir = str(image_path).rsplit("/", 1)[0]
        
        # Initialize adapter
        adapter = BreastDataAdapter(
            image_dir=image_dir,
            breast_csv=csv_path,
            wsi_ids=["NCBI785"],
            gene_ids=test_genes
        )
        
        # Get first item and test DataPoint structure
        data_point = adapter[0]
        
        assert isinstance(data_point, DataPoint)
        assert data_point.wsi_id == "NCBI785"
        assert data_point.barcode == barcode
        assert isinstance(data_point.gene_expression, dict)
        assert all(isinstance(v, float) for v in data_point.gene_expression.values())
        assert set(data_point.gene_expression.keys()) == set(sorted(test_genes))
        assert isinstance(data_point.x, (int, float))
        assert isinstance(data_point.y, (int, float))
        assert data_point.img_patch_path.endswith(f"{barcode}_NCBI785.png")
        assert os.path.exists(data_point.img_patch_path)
        
        # Test data point validation
        assert data_point.validate_datapoint("BreastDataAdapter") is True
    
    def test_getitem_missing_image(self, create_test_expression_data, temp_dir):
        """Test __getitem__ when corresponding image file is missing."""
        test_genes = ["ABCC11", "ADH1B"]
        csv_path = create_test_expression_data(gene_ids=test_genes)
        
        # Initialize adapter without creating corresponding image files
        adapter = BreastDataAdapter(
            image_dir=temp_dir,
            breast_csv=csv_path,
            wsi_ids=["NCBI785"],
            gene_ids=test_genes
        )
        
        # Should raise FileNotFoundError when trying to access data
        with pytest.raises(FileNotFoundError, match="Image patch not found"):
            adapter[0]
    
    def test_len_method(self, create_test_image, create_test_expression_data):
        """Test __len__ method returns correct length."""
        test_genes = ["ABCC11", "ADH1B"]
        num_samples = 5
        
        # Create test data
        csv_path = create_test_expression_data(num_samples=num_samples, gene_ids=test_genes)
        
        # Create images for all samples
        for i in range(num_samples):
            barcode = f"{str(i).zfill(3)}x{str(i).zfill(3)}"
            image_path = create_test_image(barcode=barcode, wsi_id="NCBI785")
        
        image_dir = str(image_path).rsplit("/", 1)[0]
        
        adapter = BreastDataAdapter(
            image_dir=image_dir,
            breast_csv=csv_path,
            wsi_ids=["NCBI785"],
            gene_ids=test_genes
        )
        
        assert len(adapter) == num_samples
    
    def test_filtering_by_wsi_ids(self, create_test_image, create_test_expression_data, temp_dir):
        """Test that adapter correctly filters data by WSI IDs."""
        test_genes = ["ABCC11", "ADH1B"]
        
        # Create CSV with multiple WSI IDs
        barcodes = [f"{str(i).zfill(3)}x{str(i).zfill(3)}" for i in range(6)]
        wsi_ids = ["NCBI785", "NCBI786", "NCBI787"]
        
        data = {
            'barcode': barcodes,
            'id': ["NCBI785", "NCBI785", "NCBI786", "NCBI786", "NCBI787", "NCBI787"],
            'x_pixel': np.random.uniform(0, 40000, 6),
            'y_pixel': np.random.uniform(0, 80000, 6),
            'combined_text': ['A focus of Cancer within Breast tissue.'] * 6,
            **{gene: np.random.uniform(0, 5, 6) for gene in test_genes}
        }
        df = pd.DataFrame(data)
        csv_path = os.path.join(temp_dir, "multi_wsi_test.csv")
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
        adapter = BreastDataAdapter(
            image_dir=image_dir,
            breast_csv=csv_path,
            wsi_ids=["NCBI785"],
            gene_ids=test_genes
        )
        
        assert len(adapter) == 2  # Only 2 samples with NCBI785
        for i in range(len(adapter)):
            data_point = adapter[i]
            assert data_point.wsi_id == "NCBI785"
        
        # Test filtering to multiple WSIs
        adapter_multi = BreastDataAdapter(
            image_dir=image_dir,
            breast_csv=csv_path,
            wsi_ids=["NCBI785", "NCBI786"],
            gene_ids=test_genes
        )
        
        assert len(adapter_multi) == 4  # 2 from each WSI
        wsi_ids_found = [adapter_multi[i].wsi_id for i in range(len(adapter_multi))]
        assert all(wsi_id in ["NCBI785", "NCBI786"] for wsi_id in wsi_ids_found)
    
    @pytest.mark.parametrize("gene_subset", [
        ["ABCC11", "ADH1B"],
        ["ABCC11", "ADH1B", "ADIPOQ", "ANKRD30A", "AQP1"],
        ["ABCC11"]
    ])
    def test_different_gene_counts(self, create_test_image, create_test_expression_data, gene_subset):
        """Test BreastDataAdapter with different numbers of genes."""
        csv_path = create_test_expression_data(gene_ids=gene_subset)
        barcode = "000x000"
        image_path = create_test_image(barcode=barcode, wsi_id="NCBI785")
        image_dir = str(image_path).rsplit("/", 1)[0]
        
        adapter = BreastDataAdapter(
            image_dir=image_dir,
            breast_csv=csv_path,
            wsi_ids=["NCBI785"],
            gene_ids=gene_subset
        )
        
        data_point = adapter[0]
        assert len(data_point.gene_expression) == len(gene_subset)
        assert set(data_point.gene_expression.keys()) == set(sorted(gene_subset))
        assert all(isinstance(v, float) for v in data_point.gene_expression.values())
    
    def test_index_out_of_bounds(self, create_test_image, create_test_expression_data):
        """Test accessing index out of bounds."""
        test_genes = ["ABCC11", "ADH1B"]
        num_samples = 3
        
        csv_path = create_test_expression_data(num_samples=num_samples, gene_ids=test_genes)
        
        for i in range(num_samples):
            barcode = f"{str(i).zfill(3)}x{str(i).zfill(3)}"
            image_path = create_test_image(barcode=barcode, wsi_id="NCBI785")
        
        image_dir = str(image_path).rsplit("/", 1)[0]
        
        adapter = BreastDataAdapter(
            image_dir=image_dir,
            breast_csv=csv_path,
            wsi_ids=["NCBI785"],
            gene_ids=test_genes
        )
        
        # Valid indices
        for i in range(num_samples):
            assert adapter[i] is not None
        
        # Invalid indices
        with pytest.raises(IndexError):
            adapter[num_samples]  # One past the end
        
        with pytest.raises(IndexError):
            adapter[-num_samples - 1]  # Too negative
    
    def test_gene_expression_values_type(self, create_test_image, create_test_expression_data):
        """Test that gene expression values are correctly converted to float."""
        test_genes = ["ABCC11", "ADH1B"]
        csv_path = create_test_expression_data(gene_ids=test_genes)
        barcode = "000x000"
        image_path = create_test_image(barcode=barcode, wsi_id="NCBI785")
        image_dir = str(image_path).rsplit("/", 1)[0]
        
        adapter = BreastDataAdapter(
            image_dir=image_dir,
            breast_csv=csv_path,
            wsi_ids=["NCBI785"],
            gene_ids=test_genes
        )
        
        data_point = adapter[0]
        
        # All gene expression values should be floats
        for gene, value in data_point.gene_expression.items():
            assert isinstance(value, float)
            assert not isinstance(value, str)
            assert not pd.isna(value)
    
    def test_coordinate_types(self, create_test_image, create_test_expression_data):
        """Test that x and y coordinates have correct types."""
        test_genes = ["ABCC11"]
        csv_path = create_test_expression_data(gene_ids=test_genes)
        barcode = "000x000"
        image_path = create_test_image(barcode=barcode, wsi_id="NCBI785")
        image_dir = str(image_path).rsplit("/", 1)[0]
        
        adapter = BreastDataAdapter(
            image_dir=image_dir,
            breast_csv=csv_path,
            wsi_ids=["NCBI785"],
            gene_ids=test_genes
        )
        
        data_point = adapter[0]
        
        # Coordinates should be numeric
        assert isinstance(data_point.x, (int, float))
        assert isinstance(data_point.y, (int, float))
        assert not pd.isna(data_point.x)
        assert not pd.isna(data_point.y)
