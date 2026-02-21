# PINN Automation Pipeline

Automated Physics-Informed Neural Network pipeline for landslide deformation prediction.

## Quick Start

```bash
# Training only (with ready CSV)
python run_pipeline.py --mode train --csv "path/to/your_data.csv" --model gru

# Full pipeline (SHP → CSV → Rainfall → Train)  
python run_pipeline.py --mode full --shp-folder "path/to/shapefiles"
```

## Project Structure

```
├── config.yaml           # All configuration
├── run_pipeline.py       # CLI entry point
├── data_pipeline/        # Data processing
├── models/               # LSTM/GRU PINN models
├── training/             # Training & tuning
├── analysis/             # Sensitivity analysis
├── visualization/        # Plots
└── outputs/              # Results, models, figures
```

## Configuration

Edit `config.yaml` to set:
- Input CSV path
- Model type (gru/lstm) and hyperparameters
- Physics parameters (Perzyna model)
- Training settings
- Checkpointing options
