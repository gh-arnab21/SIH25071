# Quick Start Guide: Specialized Rockfall Datasets

This guide will help you quickly set up and start using the specialized rockfall and mining datasets for enhanced model training.

## 🚀 Quick Setup (5 minutes)

### Step 1: Initial Setup
```bash
# Run the dataset setup script
python scripts/setup_datasets.py

# This will:
# - Create all necessary directories
# - Download available datasets (RockNet if git is available)
# - Generate download instructions for manual datasets
# - Show current status
```

### Step 2: Check Current Status
```bash
# Check what datasets are available
python scripts/setup_datasets.py --action status
```

### Step 3: Download Priority Datasets

Based on ease of integration and data quality, prioritize these datasets:

**1. Brazilian Rockfall Slope Dataset** ⭐ (Easiest - CSV format)
- Download CSV file from research paper supplementary material
- Place in: `data/raw/brazilian_rockfall_slope/slope_data.csv`
- 220 samples, 8 features, 3 risk classes

**2. Railway Rockfall Detection** ⭐ (Good for structured data)
- Contact authors or check supplementary material
- Place in: `data/raw/railway_rockfall/features.csv`
- 1,733 samples, 14 features

**3. RockNet Seismic Dataset** ⭐ (Auto-downloadable)
- Should download automatically with git
- Contains seismic data for earthquake/rockfall classification

## 📊 Validation and Testing

### Validate Downloaded Datasets
```bash
# Validate all available datasets
python scripts/validate_datasets.py

# Validate specific dataset
python scripts/validate_datasets.py --dataset brazilian

# Generate detailed report
python scripts/validate_datasets.py --report
```

### Expected Validation Output
```
🔍 Validating Brazilian Rockfall Slope...
✅ Brazilian Rockfall Slope validation completed

🔍 Validating RockNet Seismic...
✅ RockNet Seismic validation completed

VALIDATION SUMMARY
================================================================================
✅ Brazilian Rockfall Slope: ✅ PASSED
   📊 220 samples
   📈 8 features
✅ RockNet Seismic: ✅ PASSED
   📊 30 samples (limited for validation)
   📈 120 features
```

## 🏃‍♂️ Start Training

### Train on Single Dataset
```bash
# Train on Brazilian Rockfall data
python scripts/train_model.py --dataset-type brazilian_rockfall_slope --experiment-name "brazilian_slope_baseline"

# Train on Railway Rockfall data
python scripts/train_model.py --dataset-type railway_rockfall --experiment-name "railway_sensor_baseline"
```

### Train on Multiple Datasets (Fusion)
```bash
# Train on combined datasets
python scripts/train_model.py --dataset-type multi_dataset --experiment-name "multi_dataset_fusion"
```

### Monitor Training Progress
```bash
# Check experiment logs
ls experiments/

# View specific experiment results
cat experiments/brazilian_slope_baseline_*/training_log.json
```

## 📁 Expected Directory Structure After Setup

```
data/raw/
├── download_instructions.md          # Manual download guide
├── brazilian_rockfall_slope/
│   └── slope_data.csv               # Manual download needed
├── railway_rockfall/
│   └── features.csv                 # Manual download needed
├── rocknet_seismic/                 # Auto-downloaded via git
│   └── data/
│       ├── earthquake/
│       ├── rockfall/
│       └── noise/
├── open_pit_mine_detection/         # Manual download needed
│   ├── images/
│   └── annotations/
└── ...
```

## 🎯 Training Configuration Examples

### Brazilian Slope Dataset Configuration
```yaml
# config/brazilian_slope_config.yaml
dataset:
  type: "brazilian_rockfall_slope"
  path: "data/raw/brazilian_rockfall_slope"
  
model:
  type: "ensemble"
  base_models: ["random_forest", "gradient_boosting", "neural_network"]
  
training:
  validation_split: 0.2
  cross_validation_folds: 5
  random_seed: 42
```

### Multi-Dataset Fusion Configuration
```yaml
# config/multi_dataset_config.yaml
dataset:
  type: "multi_dataset"
  datasets:
    brazilian_rockfall_slope: "data/raw/brazilian_rockfall_slope"
    railway_rockfall: "data/raw/railway_rockfall"
  fusion_method: "feature_concatenation"
  
model:
  type: "ensemble"
  base_models: ["random_forest", "xgboost", "neural_network"]
```

## 🔧 Troubleshooting

### Common Issues

**1. Dataset Not Found**
```bash
# Check if directories exist
python scripts/setup_datasets.py --action status

# Recreate directories
python scripts/setup_datasets.py --action setup
```

**2. Validation Fails**
```bash
# Check specific dataset
python scripts/validate_datasets.py --dataset brazilian

# Check file contents
ls -la data/raw/brazilian_rockfall_slope/
head data/raw/brazilian_rockfall_slope/slope_data.csv
```

**3. Training Fails**
```bash
# Check dataset availability
python scripts/validate_datasets.py

# Check configuration
python scripts/train_model.py --dataset-type brazilian_rockfall_slope --dry-run
```

### Getting Help

1. **Check validation report**: `python scripts/validate_datasets.py --report`
2. **Review download instructions**: `cat data/raw/download_instructions.md`
3. **Check training logs**: `ls experiments/` and review log files
4. **Test with minimal data**: Start with Brazilian dataset (smallest, CSV format)

## 📈 Next Steps

1. **Download Priority Datasets**: Start with Brazilian Rockfall Slope Dataset
2. **Validate Setup**: Run validation script to ensure data is correctly loaded
3. **Baseline Training**: Train simple models on individual datasets
4. **Multi-Dataset Fusion**: Combine datasets for comprehensive training
5. **Hyperparameter Tuning**: Optimize models using cross-validation
6. **Production Deployment**: Save best models for inference

## 🎯 Expected Performance Baselines

Based on the datasets and typical performance:

| Dataset | Expected Accuracy | Features | Samples |
|---------|------------------|----------|---------|
| Brazilian Rockfall Slope | 85-92% | 8 geological | 220 |
| Railway Rockfall | 88-95% | 14 sensor | 1,733 |
| RockNet Seismic | 82-90% | 120 spectral | 2,000+ |
| Multi-Dataset Fusion | 90-96% | Variable | 3,000+ |

Start with individual datasets to establish baselines, then move to fusion for improved performance.