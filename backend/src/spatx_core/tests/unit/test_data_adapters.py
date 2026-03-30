"""Tests for BreastDataAdapter and Results."""
import os
import pytest
import torch
import numpy as np
from spatx_core.data_adapters import BreastDataAdapter
from spatx_core.trainers.cit_to_gene.simple_trainer import Results

def test_breast_adapter_initialization(create_test_image, create_test_expression_data):
    # Define test genes
    test_genes = ["ABCC11", "ADH1B", "ADIPOQ", "ANKRD30A", "AQP1"]
    
    # Create test data with matching image names
    csv_path = create_test_expression_data(gene_ids=test_genes)
    # Create a test image for each entry in the CSV
    for i in range(10):  # default num_samples
        barcode = f"{str(i).zfill(3)}x{str(i).zfill(3)}"
        image_path = create_test_image(barcode=barcode, wsi_id="NCBI785")
    
    # Initialize adapter
    adapter = BreastDataAdapter(
        image_dir=str(image_path).rsplit("/", 1)[0],
        breast_csv=csv_path,
        wsi_ids=["NCBI785"],
        gene_ids=test_genes
    )
    
    assert adapter is not None
    assert len(adapter) > 0
    assert adapter.gene_ids == sorted(test_genes)

def test_breast_adapter_getitem(create_test_image, create_test_expression_data):
    # Define test genes
    test_genes = ["ABCC11", "ADH1B", "ADIPOQ", "ANKRD30A", "AQP1"]
    
    # Create test data
    csv_path = create_test_expression_data(gene_ids=test_genes)
    # Create test image with known name for first item
    barcode = "000x000"  # First item in generated CSV
    image_path = create_test_image(barcode=barcode, wsi_id="NCBI785")
    
    # Initialize adapter
    adapter = BreastDataAdapter(
        image_dir=str(image_path).rsplit("/", 1)[0],
        breast_csv=csv_path,
        wsi_ids=["NCBI785"],
        gene_ids=test_genes,
    )
    
    # Get first item and test DataPoint structure
    data_point = adapter[0]
    assert data_point.wsi_id == "NCBI785"
    assert data_point.barcode == barcode
    assert isinstance(data_point.gene_expression, dict)
    assert all(isinstance(v, float) for v in data_point.gene_expression.values())
    assert set(data_point.gene_expression.keys()) == set(test_genes)
    assert isinstance(data_point.x, float)
    assert isinstance(data_point.y, float)
    assert data_point.img_patch_path.endswith(f"{barcode}_NCBI785.png")
    
    # Test data point validation
    assert data_point.validate_datapoint("BreastDataAdapter") is True

def test_breast_adapter_filtering(create_test_image, create_test_expression_data):
    # Define test genes
    test_genes = ["ABCC11", "ADH1B", "ADIPOQ", "ANKRD30A", "AQP1"]
    
    # Create test data with multiple WSIs
    csv_path = create_test_expression_data(gene_ids=test_genes)
    barcode = "000x000"
    image_path = create_test_image(barcode=barcode, wsi_id="NCBI785")
    
    # Test with specific WSI filter
    adapter = BreastDataAdapter(
        image_dir=str(image_path).rsplit("/", 1)[0],
        breast_csv=csv_path,
        wsi_ids=["NCBI785"],
        gene_ids=test_genes
    )
    
    # Check all items have correct WSI
    for i in range(len(adapter)):
        data_point = adapter[i]
        assert data_point.wsi_id == "NCBI785"
        assert isinstance(data_point.gene_expression, dict)
        assert set(data_point.gene_expression.keys()) == set(test_genes)
        assert isinstance(data_point.x, float)
        assert isinstance(data_point.y, float)
        assert data_point.img_patch_path.endswith(f"{data_point.barcode}_{data_point.wsi_id}.png")

@pytest.mark.parametrize("gene_subset", [
    ["ABCC11", "ADH1B"],
    ["ABCC11", "ADH1B", "ADIPOQ", "ANKRD30A", "AQP1"],
    ["ABCC11", "ADH1B", "ADIPOQ", "ANKRD30A", "AQP1", "AQP3", "CCR7", "CD3E", "CEACAM6", "CEACAM8"]
])
def test_breast_adapter_different_gene_counts(
    create_test_image, create_test_expression_data, gene_subset
):
    # Create test data with all possible genes
    csv_path = create_test_expression_data(gene_ids=gene_subset)
    # Create test image
    barcode = "000x000"
    image_path = create_test_image(barcode=barcode, wsi_id="NCBI785")
    
    # Initialize adapter with gene subset
    adapter = BreastDataAdapter(
        image_dir=str(image_path).rsplit("/", 1)[0],
        breast_csv=csv_path,
        wsi_ids=["NCBI785"],
        gene_ids=gene_subset
    )
    
    # Test expression dictionary length
    data_point = adapter[0]
    assert len(data_point.gene_expression) == len(gene_subset)
    assert set(data_point.gene_expression.keys()) == set(gene_subset)

def test_results_metrics_update(tmp_path):
    # Initialize Results with a temporary path
    results = Results(save_path=str(tmp_path))
    
    # Create dummy predictions and targets
    predictions = torch.randn(10, 5)  # 10 samples, 5 genes
    targets = torch.randn(10, 5)
    epoch_loss = 0.5
    
    # Test train metrics update
    results.update_metrics(0, 'train', predictions, targets, epoch_loss)
    assert len(results.train_metrics['per_gene_loss']) == 1
    assert len(results.train_metrics['per_gene_pearson']) == 1
    assert len(results.train_metrics['mean_loss']) == 1
    
    # Test validation metrics update
    results.update_metrics(0, 'val', predictions, targets, epoch_loss)
    assert len(results.val_metrics['per_gene_loss']) == 1
    assert len(results.val_metrics['per_gene_pearson']) == 1
    assert results.best_metrics['per_gene']['best_loss'] is not None

def test_results_file_saving(tmp_path):
    # Initialize Results with a temporary path
    results = Results(save_path=str(tmp_path))
    
    # Create and update metrics
    predictions = torch.randn(10, 5)
    targets = torch.randn(10, 5)
    results.update_metrics(0, 'train', predictions, targets, 0.5)
    results.update_metrics(0, 'val', predictions, targets, 0.4)
    
    # Check if files were created
    assert os.path.exists(os.path.join(tmp_path, 'train_metrics.txt'))
    assert os.path.exists(os.path.join(tmp_path, 'val_metrics.txt'))

def test_breast_adapter_invalid_input(create_test_image, create_test_expression_data):
    image_path = create_test_image()
    csv_path = create_test_expression_data()
    base_path = str(image_path).rsplit("/", 1)[0]

def test_device_handling(create_test_image, create_test_expression_data, device):
    from spatx_core.trainers.cit_to_gene.simple_trainer import SimpleCITTrainer

def test_end_to_end_training(create_test_image, create_test_expression_data, tmp_path, device):
    from spatx_core.trainers.cit_to_gene.simple_trainer import SimpleCITTrainer
    
    # Define a small set of genes for quick testing
    test_genes = ["ABCC11", "ADH1B", "ADIPOQ", "ANKRD30A", "AQP1"]
    
    # Create test data with multiple samples
    csv_path = create_test_expression_data(num_samples=4, gene_ids=test_genes)  # Small batch size
    
    # Create images for all samples
    image_dir = None
    for i in range(4):
        barcode = f"{str(i).zfill(3)}x{str(i).zfill(3)}"
        image_path = create_test_image(barcode=barcode, wsi_id="NCBI785")
        if image_dir is None:
            image_dir = str(image_path).rsplit("/", 1)[0]
    base_path = str(image_path).rsplit("/", 1)[0]
    
    # Initialize adapters
    train_adapter = BreastDataAdapter(
        image_dir=str(image_path).rsplit("/", 1)[0],
        breast_csv=csv_path,
        wsi_ids=["NCBI785"],
        gene_ids=test_genes
    )
    
    val_adapter = BreastDataAdapter(
        image_dir=str(image_path).rsplit("/", 1)[0],
        breast_csv=csv_path,
        wsi_ids=["NCBI785"],
        gene_ids=test_genes
    )
    
    # Initialize trainer
    trainer = SimpleCITTrainer(
        train_adapter=train_adapter,
        validation_adapter=val_adapter,
        num_epochs=2,  # Small number of epochs for testing
        batch_size=2,
        device='cpu',  # Use CPU for testing
        results_path=str(tmp_path)
    )
    
    # Train model
    model, results = trainer.train()
    
    # Basic assertions to verify training completed
    assert model is not None
    assert results is not None
    assert len(results.train_metrics['mean_loss']) == 2  # 2 epochs
    assert len(results.val_metrics['mean_loss']) == 2
    assert os.path.exists(os.path.join(tmp_path, 'train_metrics.txt'))
    assert os.path.exists(os.path.join(tmp_path, 'val_metrics.txt'))
    
    
    # Test with non-existent WSI ID
    with pytest.raises(ValueError):
        BreastDataAdapter(
            image_dir=base_path,
            breast_csv=csv_path,
            wsi_ids=["NONEXISTENT"],
            gene_ids=test_genes,
        )
    
    # Test with non-existent image directory
    with pytest.raises(ValueError):
        BreastDataAdapter(
            image_dir="/nonexistent/path",
            breast_csv=csv_path,
            wsi_ids=["NCBI785"],
            gene_ids=test_genes,
        )
    
    # Test with non-existent CSV file
    with pytest.raises(FileNotFoundError):
        BreastDataAdapter(
            image_dir=base_path,
            breast_csv="/nonexistent/file.csv",
            wsi_ids=["NCBI785"],
            gene_ids=test_genes,
        )
