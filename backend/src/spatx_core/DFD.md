# Data Flow Diagram for SpatX Training Pipeline

## 1. Data Sources
```mermaid
flowchart TD
    subgraph External Data Sources
        ImgDir[20x Image Directory<br>*.png files] --> |Raw Images| BA
        CSV[breast.csv<br>Gene Expression Data] --> |Expression Values| BA
        WSI_IDs[WSI IDs Lists<br>Train/Val Split] --> |Data Split Config| BA
    end

    subgraph Data Adapters
        BA[BreastDataAdapter]
        BA --> |get_items()| Data1[Data Class<br>train_data]
        BA --> |get_items()| Data2[Data Class<br>validation_data]
    end

    subgraph Dataset Processing
        Data1 --> |__getitem__| DS1[CITDataset<br>train_dataset]
        Data2 --> |__getitem__| DS2[CITDataset<br>validation_dataset]
        DS1 --> |DataLoader| DL1[train_dataloader<br>batch_size=8<br>shuffle=True]
        DS2 --> |DataLoader| DL2[validation_dataloader<br>batch_size=8<br>shuffle=False]
    end
```

## 2. Model Architecture & Training
```mermaid
flowchart TD
    subgraph Data Flow
        Train_Adp[Train Adapter<br>TENX99,TENX95,<br>NCBI783,NCBI784] --> |get_items| TrainD[Train Data]
        Val_Adp[Validation Adapter<br>NCBI785] --> |get_items| ValD[Validation Data]
        TrainD --> |CITDataset| DL1[Train DataLoader<br>batch_size=8<br>shuffle=True]
        ValD --> |CITDataset| DL2[Val DataLoader<br>batch_size=8<br>shuffle=False]
    end

    subgraph Model Components
        CIT[CIT Backbone<br>Vision Transformer] --> |Feature Extraction| CITG[CITGenePredictor]
        CITG --> |Forward Pass| Pred[Predictions<br>num_genes dim]
    end

    subgraph Training Loop
        DL1 --> |Training Batch| TProc[Training Process]
        TProc --> |Images + Labels| CITG
        CITG --> |Predictions| Loss[CombinedLoss<br>alpha=0.5, reg=1.0]
        Loss --> |Gradient| Opt[Adam Optimizer<br>lr=0.001]
        Opt --> |Update| CITG
        
        DL2 --> |Validation Batch| VProc[Validation Process]
        VProc --> |Images + Labels| CITG
        CITG --> |Val Predictions| Loss
    end

    subgraph Metrics Tracking
        Loss --> |epoch_loss| Res[Results Class]
        Pred --> |predictions| Res
        Labels[Ground Truth] --> |targets| Res
        Res --> |update_metrics| MT[Metrics Tracking<br>- per_gene_loss<br>- per_gene_pearson<br>- mean_loss<br>- mean_pearson<br>- loss_variance<br>- pearson_variance]
        
        MT --> |Training Metrics| Train_MT[Train Metrics<br>Per Epoch]
        MT --> |Validation Metrics| Val_MT[Validation Metrics<br>Per Epoch]
        Val_MT --> |Compare| Best[Best Model State<br>Lowest Val Loss]
    end
```

## 3. Output & Storage
```mermaid
flowchart TD
    subgraph Storage Management
        Exp[Experiment Directory<br>/runs/experiment1/] --> |Save| Model[model.pth<br>Best Model State]
        Exp --> |Save| Metrics[metrics/<br>- train_metrics.txt<br>- val_metrics.txt]
    end

    subgraph Results Content
        Metrics --> |Contains| BM[Best Metrics<br>- Best Mean Loss<br>- Best Mean Pearson<br>- Best Epoch]
        Metrics --> |Contains| PM[Per-Gene Metrics<br>- Loss<br>- Pearson Correlation]
        Metrics --> |Contains| TM[Training History<br>- Per-epoch metrics<br>- Loss variance<br>- Pearson variance]
    end
```

## Key Data Flow Steps:

1. **Data Ingestion**:
   - Images (*.png) → BreastDataAdapter
   - Gene expressions (CSV) → BreastDataAdapter
   - WSI IDs for train/val split → Separate adapters

2. **Data Processing**:
   - BreastDataAdapter → Data class → CITDataset → DataLoader
   - Outputs batches of (images, expressions, metadata)

3. **Model Flow**:
   - Images → CIT Backbone → Feature extraction
   - Features → CITGenePredictor → Gene expression predictions
   - Predictions + Ground truth → CombinedLoss → Loss value

4. **Training Process**:
   - Loss → Backward pass → Gradient computation
   - Optimizer → Model parameter updates
   - Validation → Best model state saving

5. **Metrics Collection**:
   - Predictions + Ground truth → Results class
   - Per-gene metrics computation
   - Best metrics tracking
   - Metrics file writing

6. **Output Storage**:
   - Model state → model.pth
   - Training metrics → metrics/*.txt
   - All saved in experiment-specific directory

## Key Variables:
- `train_adp`, `validation_adp`: BreastDataAdapter instances
- `train_data`, `validation_data`: Data class instances
- `train_dataset`, `validation_dataset`: CITDataset instances
- `model`: CITGenePredictor instance
- `optimizer`: Adam optimizer
- `criterion`: CombinedLoss instance
- `results`: Results tracking instance
- `best_model_state`: Best performing model state
- `best_val_loss`: Best validation loss achieved
