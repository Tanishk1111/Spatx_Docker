# SpatX Core

A modular Python library for spatial transcriptomics data processing and deep learning models.

## Overview

SpatX Core provides a flexible architecture for building spatial transcriptomics analysis pipelines. It follows a modular design pattern that separates data handling, model development, and training logic.

## Architecture

### Core Components

1. **Data Layer** (`core/data/`)
   - `Data`: Base class that standardizes spatial transcriptomics data format
   - `DataPoint`: Container for single spatial transcriptomics observation

2. **Data Adapters** (`core/data_adapters/`)
   - `BaseDataAdapter`: Abstract interface for data source implementations
   - `BreastDataAdapter`: Implementation for breast cancer spatial data
   - Custom adapters can be created for new data sources

3. **Datasets** (`core/datasets/`)
   - Model-specific dataset implementations
   - `CITDataset`: Dataset for CIT-to-Gene prediction model
   - Handles data transformations and tensor conversions

4. **Models** (`core/models/`)
   - Neural network model implementations
   - Current models:
     - `CIT`: Compact vision transformer backbone
     - `CITGene`: Gene expression predictor using CIT features

5. **Trainers** (`core/trainers/`)
   - Training loop implementations
   - `SimpleCITTrainer`: Basic trainer for CIT-to-Gene model

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/spatx.git

# Install dependencies
cd spatx/spatx_core
pip install -e .
```

## Quick Start

```python
from spatx_core.data_adapters import BreastDataAdapter
from spatx_core.trainers import SimpleCITTrainer

# Create data adapters
train_adapter = BreastDataAdapter(
    image_dir="path/to/images",
    breast_csv="path/to/train.csv"
)
val_adapter = BreastDataAdapter(
    image_dir="path/to/images",
    breast_csv="path/to/val.csv"
)

# Initialize trainer
trainer = SimpleCITTrainer(
    train_data_adapter=train_adapter,
    validation_data_adapter=val_adapter,
    num_genes=50,
    batch_size=16,
    device='cuda'
)

# Train model
model = trainer.train()
```

## Extending the Library

### Adding New Data Sources

1. Create a new adapter in `core/data_adapters/`:
```python
from spatx_core.data_adapters import BaseDataAdapter

class NewDataAdapter(BaseDataAdapter):
    def __init__(self, ...):
        # Initialize your data source
        pass

    def __getitem__(self, idx):
        # Return standardized DataPoint
        pass

    def __len__(self):
        # Return number of samples
        pass
```

### Adding New Models

1. Create model architecture in `core/models/your_model/`
2. Create corresponding dataset in `core/datasets/your_model/`
3. Implement trainer in `core/trainers/your_model/`

### Best Practices

1. **Data Adapters**
   - Implement data loading logic
   - Convert to standard DataPoint format
   - Handle file paths and basic preprocessing

2. **Datasets**
   - Implement model-specific transformations
   - Convert data to appropriate tensor formats
   - Handle augmentations if needed

3. **Models**
   - Keep architecture in separate files
   - Include docstrings with expected input formats
   - Document output shapes and types

4. **Trainers**
   - Handle device management
   - Implement training/validation loops
   - Track best model states

## Project Structure
```
spatx_core/
├── core/
│   ├── data/
│   │   ├── __init__.py
│   │   └── data.py
│   ├── data_adapters/
│   │   ├── __init__.py
│   │   ├── base_data_adapter.py
│   │   └── breast_csv.py
│   ├── datasets/
│   │   └── cit_to_gene/
│   │       └── dataset.py
│   ├── models/
│   │   └── cit_to_gene/
│   │       ├── CiT_Net_T.py
│   │       └── CiTGene.py
│   └── trainers/
│       └── cit_to_gene/
│           └── simple_trainer.py
├── pyproject.toml
└── README.md
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Implement your feature/fix
4. Add tests if applicable
5. Submit a pull request

## License

[Add your license here]

## Citation

[Add citation information if applicable]