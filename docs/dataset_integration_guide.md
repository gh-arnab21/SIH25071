# Dataset Integration Guide

This guide provides step-by-step instructions for downloading and integrating the collected datasets into the Rockfall Prediction System.

## Dataset Overview

The system supports 10 specialized datasets covering different aspects of rockfall and mining safety:

### 🏭 Mining-Focused Datasets
1. **Open Pit Mine Object Detection Dataset** (1.06 GB)
2. **Open-Pit Mine Segmentation Dataset** (179.13 MB)
3. **Mine Safety Performance Dataset (Queensland)**

### 🔬 Specialized Rockfall Datasets
4. **RockNet Seismic Dataset**
5. **Railway Rockfall Detection Dataset**
6. **Underground Mining Face Dataset (DsLMF+)**

### 🌍 Environmental and Geographic Data
7. **USGS Digital Elevation Data**
8. **Environmental Sensor Data (Global)**

### 📊 Ready-to-Use Structured Datasets
9. **Brazilian Rockfall Slope Dataset**
10. **Weather APIs for Real-Time Data**

## Download Instructions

### 1. Open Pit Mine Object Detection Dataset

```bash
# Create dataset directory
mkdir -p data/raw/open_pit_mine_detection

# Download manually from:
# https://figshare.com/articles/dataset/Open_Pit_Mine_Object_Detection_Dataset/27300960
# Extract to: data/raw/open_pit_mine_detection/

# Expected structure:
# data/raw/open_pit_mine_detection/
# ├── images/
# │   ├── image_001.jpg
# │   └── ...
# └── annotations/
#     ├── annotations.json
#     └── ...
```

### 2. Open-Pit Mine Segmentation Dataset

```bash
# Create dataset directory
mkdir -p data/raw/open_pit_mine_segmentation

# Download manually from:
# https://figshare.com/articles/dataset/An_open-pit_mine_segmentation_dataset_for_deep_learning/27301734
# Extract to: data/raw/open_pit_mine_segmentation/

# Expected structure:
# data/raw/open_pit_mine_segmentation/
# ├── images/
# ├── labels/          # YOLO format annotations
# └── classes.txt
```

### 3. Mine Safety Performance Dataset (Queensland)

```bash
# Create dataset directory
mkdir -p data/raw/queensland_mine_safety

# Download manually from:
# https://www.data.qld.gov.au/dataset/mine-site-safety-performance
# Save XLS files to: data/raw/queensland_mine_safety/

# Expected files:
# - surface_mines_2018-2024.xls
# - underground_mines_2018-2024.xls
# - quarries_2018-2024.xls
```

### 4. RockNet Seismic Dataset

```bash
# Create dataset directory
mkdir -p data/raw/rocknet_seismic

# Method 1: Git clone
git clone https://github.com/tso1257771/RockNet.git data/raw/rocknet_seismic/

# Method 2: Manual download from Dryad
# https://doi.org/10.5061/dryad.tx95x6b2f
# Extract to: data/raw/rocknet_seismic/

# Expected structure:
# data/raw/rocknet_seismic/
# ├── data/
# │   ├── earthquake/
# │   ├── rockfall/
# │   └── noise/
# └── scripts/
```

### 5. Railway Rockfall Detection Dataset

```bash
# Create dataset directory
mkdir -p data/raw/railway_rockfall

# Download manually (contact authors or check associated papers)
# Expected structure:
# data/raw/railway_rockfall/
# ├── features.csv     # 1,733 samples with 14 features
# ├── labels.csv
# └── metadata.json
```

### 6. Underground Mining Face Dataset (DsLMF+)

```bash
# Create dataset directory
mkdir -p data/raw/underground_mining_face

# Download manually from figshare (linked in Nature paper)
# https://www.nature.com/articles/s41597-023-02322-9
# Extract to: data/raw/underground_mining_face/

# Expected structure:
# data/raw/underground_mining_face/
# ├── images/          # 138,004 images
# ├── annotations/     # YOLO/COCO format
# └── classes.txt
```

### 7. USGS Digital Elevation Data

```bash
# Create dataset directory
mkdir -p data/raw/usgs_elevation

# Download manually from:
# https://www.usgs.gov/the-national-map-data-delivery/gis-data-download
# Select area of interest and download GeoTIFF files
# Extract to: data/raw/usgs_elevation/

# Expected structure:
# data/raw/usgs_elevation/
# ├── region_1/
# │   ├── dem_1m.tif
# │   └── metadata.xml
# └── region_2/
```

### 8. Environmental Sensor Data (Global)

```bash
# Create dataset directory
mkdir -p data/raw/environmental_sensors

# Clone repository for processing tools
git clone https://github.com/opendata-stuttgart/sensors-software.git data/raw/environmental_sensors/tools/

# Download sensor data via their APIs or S3 buckets
# Save CSV files to: data/raw/environmental_sensors/data/
```

### 9. Brazilian Rockfall Slope Dataset

```bash
# Create dataset directory
mkdir -p data/raw/brazilian_rockfall_slope

# This dataset should be available as a CSV file
# If from a paper, check supplementary materials
# Save as: data/raw/brazilian_rockfall_slope/slope_data.csv

# Expected columns:
# - Sample ID, Rock Type, Slope Angle, Joint Spacing, 
# - Water Presence, Weathering, etc. (8 variables scored 1-4)
# - Risk Score (8-32 range)
```

### 10. Weather APIs Setup

```bash
# Create configuration for weather APIs
mkdir -p data/raw/weather_apis

# Create API configuration file
cat > data/raw/weather_apis/config.yaml << EOF
noaa:
  base_url: "https://api.weather.gov"
  documentation: "https://www.weather.gov/documentation/services-web-api"
  
openweathermap:
  base_url: "https://api.openweathermap.org/data/2.5"
  api_key: "YOUR_API_KEY_HERE"  # Sign up at openweathermap.org
EOF
```

## Post-Download Setup

After downloading all datasets, run the integration setup:

```bash
# Run the dataset integration script
python scripts/setup_datasets.py

# Verify dataset structure
python scripts/validate_datasets.py

# Generate dataset statistics
python scripts/analyze_datasets.py
```

## Dataset Integration with Training Script

Once datasets are downloaded, you can use them with the training script:

```bash
# Train on specific dataset
python scripts/train_model.py \
    --experiment-name rocknet_training \
    --dataset-type rocknet_seismic \
    --data-dir data/raw/rocknet_seismic

# Train on Brazilian slope data
python scripts/train_model.py \
    --experiment-name slope_stability \
    --dataset-type brazilian_rockfall \
    --data-dir data/raw/brazilian_rockfall_slope

# Multi-dataset training
python scripts/train_model.py \
    --experiment-name multimodal_training \
    --dataset-type multimodal \
    --config config/multimodal_config.yaml
```

## Data Preprocessing

Each dataset requires specific preprocessing:

1. **Image Datasets**: Resize, normalize, augment
2. **Seismic Data**: Filter, segment, feature extraction
3. **Sensor Data**: Time-series processing, anomaly detection
4. **Structured Data**: Feature engineering, scaling
5. **Geographic Data**: Coordinate transformation, terrain analysis

## Next Steps

1. Download the datasets you're most interested in first
2. Start with the Brazilian Rockfall Slope Dataset (structured data, easiest to integrate)
3. Then move to RockNet Seismic Dataset (well-documented)
4. Finally integrate the larger image datasets

Let me know which datasets you'd like to prioritize, and I can create specific data loaders for them!