# Mining Datasets Preprocessing Report

**Generated:** 2025-09-19 23:51:49.107300

## Processed Datasets

### mining_combined_processed

- **Samples:** 400
- **Features:** 41
- **Classes:** 2
- **Class Distribution:** {1: 178, 2: 222}
- **File Size:** 0.16 MB

### object_detection_processed

- **Samples:** 200
- **Features:** 37
- **Classes:** 2
- **Class Distribution:** {1: 135, 2: 65}
- **File Size:** 0.07 MB

### segmentation_processed

- **Samples:** 200
- **Features:** 41
- **Classes:** 2
- **Class Distribution:** {1: 43, 2: 157}
- **File Size:** 0.08 MB

## Feature Descriptions

### Image Features (21 features)
1. Gray statistics: mean, std, min, max
2. RGB statistics: means and stds for R, G, B channels
3. Texture features: gradient magnitude and directional gradients
4. Edge density
5. HSV color space statistics

### Object Detection Features (16 features)
1. Number of detected objects
2. Object size statistics: area mean, std, min, max
3. Aspect ratio statistics
4. Object distribution: spatial spread and centers
5. Unique object classes

### Segmentation Features (20 features)
1. Number of segments
2. Segment complexity statistics
3. Class diversity metrics
4. Coordinate distribution analysis
5. Coverage area metrics

## Risk Classification

**Object Detection Risk Levels:**
- Level 0 (Low): No objects detected
- Level 1 (Medium): 1-2 objects detected
- Level 2 (High): 3+ objects detected

**Segmentation Risk Levels:**
- Level 0 (Low): Simple segmentation (complexity < 20)
- Level 1 (Medium): Moderate complexity (20-100)
- Level 2 (High): High complexity (100+)

