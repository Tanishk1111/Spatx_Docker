"""Tests for BaseDataAdapter abstract class."""
import pytest
from abc import ABC

from spatx_core.data_adapters.base_data_adapter import BaseDataAdapter
from spatx_core.data.data_point import DataPoint


class ConcreteDataAdapter(BaseDataAdapter):
    """Concrete implementation of BaseDataAdapter for testing."""
    
    name = "ConcreteDataAdapter"
    
    def __init__(self, gene_ids, data_points=None):
        self.gene_ids = gene_ids
        self.data_points = data_points or []
    
    def __getitem__(self, idx):
        return self.data_points[idx]
    
    def __len__(self):
        return len(self.data_points)


class TestBaseDataAdapter:
    """Test cases for BaseDataAdapter abstract class."""
    
    def test_base_adapter_is_abstract(self):
        """Test that BaseDataAdapter is an abstract class."""
        assert issubclass(BaseDataAdapter, ABC)
        
        # Should not be able to instantiate BaseDataAdapter directly
        with pytest.raises(TypeError):
            BaseDataAdapter(gene_ids=["GENE1"])  # type: ignore
    
    def test_concrete_implementation(self, temp_dir):
        """Test that concrete implementation works correctly."""
        gene_ids = ["GENE1", "GENE2"]
        adapter = ConcreteDataAdapter(gene_ids=gene_ids)
        
        assert adapter.gene_ids == gene_ids
        assert adapter.name == "ConcreteDataAdapter"
        assert len(adapter) == 0
    
    def test_concrete_implementation_with_data(self, temp_dir):
        """Test concrete implementation with actual data points."""
        from PIL import Image
        import numpy as np
        import os
        
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
        
        gene_ids = ["GENE1", "GENE2"]
        adapter = ConcreteDataAdapter(gene_ids=gene_ids, data_points=[datapoint])
        
        assert len(adapter) == 1
        assert adapter[0] == datapoint
        assert adapter.gene_ids == gene_ids
    
    def test_abstract_methods_required(self):
        """Test that all abstract methods must be implemented."""
        # This class is missing implementations of abstract methods
        class IncompleteAdapter(BaseDataAdapter):
            name = "IncompleteAdapter"
            
            def __init__(self, gene_ids):
                self.gene_ids = gene_ids
        
        # Should raise TypeError when trying to instantiate
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            IncompleteAdapter(gene_ids=["GENE1"])  # type: ignore
    
    def test_name_attribute(self):
        """Test that concrete implementations have name attribute."""
        gene_ids = ["GENE1"]
        adapter = ConcreteDataAdapter(gene_ids=gene_ids)
        
        assert hasattr(adapter, 'name')
        assert isinstance(adapter.name, str)
        assert adapter.name == "ConcreteDataAdapter"
