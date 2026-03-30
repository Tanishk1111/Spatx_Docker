"""Common test fixtures and utilities."""
import os
import pytest
import torch
import numpy as np
from PIL import Image
import pandas as pd
import tempfile
import shutil

# Test configuration
TEST_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

@pytest.fixture(scope="session")
def test_data_dir():
    """Return path to test data directory."""
    return os.path.join(os.path.dirname(__file__), "test_data")

@pytest.fixture(scope="session")
def create_test_image(test_data_dir):
    """Create a test image and save it."""
    os.makedirs(os.path.join(test_data_dir, "images"), exist_ok=True)
    
    def _create_image(barcode="000x000", wsi_id="NCBI785", size=(224, 224)):
        # Create random image
        img = np.random.randint(0, 255, (*size, 3), dtype=np.uint8)
        img = Image.fromarray(img)
        
        # Create image name in correct format
        img_name = f"{barcode}_{wsi_id}.png"
        
        # Save image
        path = os.path.join(test_data_dir, "images", img_name)
        img.save(path)
        return path
    
    return _create_image

@pytest.fixture(scope="session")
def create_test_expression_data(test_data_dir):
    """Create test expression data CSV."""
    os.makedirs(os.path.join(test_data_dir, "expression"), exist_ok=True)
    
    def _create_csv(num_samples=10, gene_ids=None):
        if gene_ids is None:
            gene_ids = ["ABCC11", "ADH1B", "ADIPOQ", "ANKRD30A", "AQP1"]  # default test genes
        
        # Generate barcodes in correct format (e.g., 000x000, 000x001, etc.)
        barcodes = [f"{str(i).zfill(3)}x{str(i).zfill(3)}" for i in range(num_samples)]
        
        # Create random expression data
        data = {
            'barcode': barcodes,
            'id': ['NCBI785'] * num_samples,  # match the column name in real data
            'x_pixel': np.random.uniform(0, 40000, num_samples),
            'y_pixel': np.random.uniform(0, 80000, num_samples),
            'combined_text': ['A focus of Cancer within Breast tissue.'] * num_samples,
            **{gene: np.random.uniform(0, 5, num_samples) for gene in gene_ids}
        }
        df = pd.DataFrame(data)
        
        # Save CSV
        path = os.path.join(test_data_dir, "expression", "test_expression.csv")
        df.to_csv(path, index=False)
        return path
    
    return _create_csv

@pytest.fixture
def device():
    """Return compute device - configurable for CPU/GPU testing."""
    return TEST_DEVICE

@pytest.fixture
def dummy123_gene_ids():
    """Return gene IDs from dummy123 model for integration tests."""
    from spatx_core.saved_models.cit_to_gene.dummy123 import gene_ids
    return gene_ids

@pytest.fixture
def temp_dir():
    """Create a temporary directory for test outputs."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)

@pytest.fixture
def sample_model_config():
    """Return a sample model configuration for testing."""
    return {
        'img_size': 224,
        'in_chans': 3,
        'embed_dim': 96,
        'depths': [2, 2, 6, 2],
        'num_heads': [3, 6, 12, 24],
    }

@pytest.fixture(scope="session")
def create_prediction_data(test_data_dir):
    """Create test data for prediction (without gene expression labels)."""
    os.makedirs(os.path.join(test_data_dir, "prediction"), exist_ok=True)
    
    def _create_prediction_csv(num_samples=5):
        # Generate barcodes 
        barcodes = [f"{str(i).zfill(3)}x{str(i).zfill(3)}" for i in range(num_samples)]
        
        # Create prediction data (no gene expression)
        data = {
            'barcode': barcodes,
            'id': ['NCBI785'] * num_samples,
            'x_pixel': np.random.uniform(0, 40000, num_samples),
            'y_pixel': np.random.uniform(0, 80000, num_samples),
            'combined_text': ['A focus of Cancer within Breast tissue.'] * num_samples,
        }
        df = pd.DataFrame(data)
        
        # Save CSV
        path = os.path.join(test_data_dir, "prediction", "test_prediction.csv")
        df.to_csv(path, index=False)
        return path
    
    return _create_prediction_csv
