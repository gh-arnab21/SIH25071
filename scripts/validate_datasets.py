#!/usr/bin/env python3
"""
Dataset validation script for specialized rockfall datasets.
"""

import os
import sys
import logging
import traceback
from pathlib import Path
from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.specialized_loaders import (
    BrazilianRockfallSlopeLoader,
    RockNetSeismicLoader,
    OpenPitMineDetectionLoader,
    RailwayRockfallLoader,
    MultiDatasetLoader
)


class DatasetValidator:
    """Validates specialized rockfall datasets."""
    
    def __init__(self, base_data_dir: str = "data/raw"):
        self.base_data_dir = Path(base_data_dir)
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        self.validation_results = {}
    
    def validate_brazilian_rockfall(self) -> Dict[str, Any]:
        """Validate Brazilian Rockfall Slope Dataset."""
        dataset_name = "Brazilian Rockfall Slope"
        print(f"\n🔍 Validating {dataset_name}...")
        
        try:
            dataset_dir = self.base_data_dir / "brazilian_rockfall_slope"
            loader = BrazilianRockfallSlopeLoader(str(dataset_dir))
            
            # Load data
            X, y, metadata = loader.load_data()
            
            # Validation checks
            results = {
                "status": "✅ PASSED",
                "dataset_name": dataset_name,
                "num_samples": len(X),
                "num_features": X.shape[1] if X.ndim > 1 else 1,
                "num_classes": len(np.unique(y)),
                "class_distribution": dict(zip(*np.unique(y, return_counts=True))),
                "feature_names": metadata.get('feature_names', []),
                "data_types": {
                    "X_dtype": str(X.dtype),
                    "y_dtype": str(y.dtype),
                    "X_shape": X.shape,
                    "y_shape": y.shape
                },
                "checks": []
            }
            
            # Data quality checks
            if X.shape[0] != y.shape[0]:
                results["checks"].append("❌ X and y have different number of samples")
                results["status"] = "❌ FAILED"
            else:
                results["checks"].append("✅ X and y sample count matches")
            
            # Check for missing values
            if np.isnan(X).any():
                nan_count = np.isnan(X).sum()
                results["checks"].append(f"⚠️  {nan_count} missing values in features")
            else:
                results["checks"].append("✅ No missing values in features")
            
            # Check feature variance
            if X.ndim > 1:
                low_variance_features = np.var(X, axis=0) < 1e-8
                if low_variance_features.any():
                    results["checks"].append(f"⚠️  {low_variance_features.sum()} low variance features")
                else:
                    results["checks"].append("✅ All features have sufficient variance")
            
            print(f"✅ {dataset_name} validation completed")
            
        except Exception as e:
            results = {
                "status": "❌ FAILED",
                "dataset_name": dataset_name,
                "error": str(e),
                "traceback": traceback.format_exc()
            }
            print(f"❌ {dataset_name} validation failed: {e}")
        
        return results
    
    def validate_rocknet_seismic(self) -> Dict[str, Any]:
        """Validate RockNet Seismic Dataset."""
        dataset_name = "RockNet Seismic"
        print(f"\n🔍 Validating {dataset_name}...")
        
        try:
            dataset_dir = self.base_data_dir / "rocknet_seismic"
            
            # Check if directory structure exists
            expected_dirs = ['data/earthquake', 'data/rockfall', 'data/noise']
            missing_dirs = []
            for dir_path in expected_dirs:
                if not (dataset_dir / dir_path).exists():
                    missing_dirs.append(dir_path)
            
            if missing_dirs:
                results = {
                    "status": "❌ FAILED",
                    "dataset_name": dataset_name,
                    "error": f"Missing directories: {missing_dirs}",
                    "available_dirs": [str(p.relative_to(dataset_dir)) 
                                     for p in dataset_dir.rglob('*') if p.is_dir()]
                }
                print(f"❌ {dataset_name} validation failed: Missing directories")
                return results
            
            loader = RockNetSeismicLoader(str(dataset_dir))
            
            # Load sample data (limit for validation)
            X, y, metadata = loader.load_data(max_samples_per_class=10)
            
            results = {
                "status": "✅ PASSED",
                "dataset_name": dataset_name,
                "num_samples": len(X),
                "num_features": X.shape[1] if X.ndim > 1 else 1,
                "num_classes": len(np.unique(y)),
                "class_distribution": dict(zip(*np.unique(y, return_counts=True))),
                "sampling_rate": metadata.get('sampling_rate'),
                "feature_extraction": metadata.get('feature_extraction_method'),
                "data_types": {
                    "X_dtype": str(X.dtype),
                    "y_dtype": str(y.dtype),
                    "X_shape": X.shape,
                    "y_shape": y.shape
                },
                "checks": []
            }
            
            # Basic validation checks
            if X.shape[0] != y.shape[0]:
                results["checks"].append("❌ X and y have different number of samples")
                results["status"] = "❌ FAILED"
            else:
                results["checks"].append("✅ X and y sample count matches")
            
            # Check for NaN values
            if np.isnan(X).any():
                results["checks"].append(f"⚠️  Contains NaN values")
            else:
                results["checks"].append("✅ No NaN values")
            
            # Check file count in each directory
            for dir_path in expected_dirs:
                full_path = dataset_dir / dir_path
                file_count = len(list(full_path.glob('*.SAC')))
                results["checks"].append(f"📁 {dir_path}: {file_count} SAC files")
            
            print(f"✅ {dataset_name} validation completed")
            
        except Exception as e:
            results = {
                "status": "❌ FAILED",
                "dataset_name": dataset_name,
                "error": str(e),
                "traceback": traceback.format_exc()
            }
            print(f"❌ {dataset_name} validation failed: {e}")
        
        return results
    
    def validate_open_pit_mine(self) -> Dict[str, Any]:
        """Validate Open Pit Mine Detection Dataset."""
        dataset_name = "Open Pit Mine Detection"
        print(f"\n🔍 Validating {dataset_name}...")
        
        try:
            dataset_dir = self.base_data_dir / "open_pit_mine_detection"
            loader = OpenPitMineDetectionLoader(str(dataset_dir))
            
            # Load sample data
            X, y, metadata = loader.load_data(max_samples=20)
            
            results = {
                "status": "✅ PASSED",
                "dataset_name": dataset_name,
                "num_samples": len(X),
                "image_shape": X[0].shape if len(X) > 0 else None,
                "num_annotations": len(y),
                "annotation_format": type(y[0]).__name__ if len(y) > 0 else None,
                "metadata": metadata,
                "checks": []
            }
            
            # Check data consistency
            if len(X) != len(y):
                results["checks"].append("❌ Mismatch between images and annotations")
                results["status"] = "❌ FAILED"
            else:
                results["checks"].append("✅ Images and annotations count matches")
            
            # Check image properties
            if len(X) > 0:
                img_shapes = [img.shape for img in X[:5]]  # Check first 5 images
                unique_shapes = set(img_shapes)
                if len(unique_shapes) > 1:
                    results["checks"].append(f"⚠️  Variable image shapes: {unique_shapes}")
                else:
                    results["checks"].append(f"✅ Consistent image shape: {img_shapes[0]}")
            
            # Check annotation structure
            if len(y) > 0:
                annotation_keys = set()
                for ann in y[:5]:  # Check first 5 annotations
                    if isinstance(ann, dict):
                        annotation_keys.update(ann.keys())
                results["checks"].append(f"📋 Annotation keys: {list(annotation_keys)}")
            
            print(f"✅ {dataset_name} validation completed")
            
        except Exception as e:
            results = {
                "status": "❌ FAILED",
                "dataset_name": dataset_name,
                "error": str(e),
                "traceback": traceback.format_exc()
            }
            print(f"❌ {dataset_name} validation failed: {e}")
        
        return results
    
    def validate_railway_rockfall(self) -> Dict[str, Any]:
        """Validate Railway Rockfall Detection Dataset."""
        dataset_name = "Railway Rockfall Detection"
        print(f"\n🔍 Validating {dataset_name}...")
        
        try:
            dataset_dir = self.base_data_dir / "railway_rockfall"
            loader = RailwayRockfallLoader(str(dataset_dir))
            
            # Load data
            X, y, metadata = loader.load_data()
            
            results = {
                "status": "✅ PASSED",
                "dataset_name": dataset_name,
                "num_samples": len(X),
                "num_features": X.shape[1] if X.ndim > 1 else 1,
                "num_classes": len(np.unique(y)),
                "class_distribution": dict(zip(*np.unique(y, return_counts=True))),
                "feature_names": metadata.get('feature_names', []),
                "data_types": {
                    "X_dtype": str(X.dtype),
                    "y_dtype": str(y.dtype),
                    "X_shape": X.shape,
                    "y_shape": y.shape
                },
                "checks": []
            }
            
            # Validation checks
            if X.shape[0] != y.shape[0]:
                results["checks"].append("❌ X and y have different number of samples")
                results["status"] = "❌ FAILED"
            else:
                results["checks"].append("✅ X and y sample count matches")
            
            # Check expected sample count
            expected_samples = 1733
            if len(X) != expected_samples:
                results["checks"].append(f"⚠️  Expected {expected_samples} samples, got {len(X)}")
            else:
                results["checks"].append(f"✅ Expected sample count: {expected_samples}")
            
            # Check expected feature count
            expected_features = 14
            if X.shape[1] != expected_features:
                results["checks"].append(f"⚠️  Expected {expected_features} features, got {X.shape[1]}")
            else:
                results["checks"].append(f"✅ Expected feature count: {expected_features}")
            
            # Check for missing values
            if np.isnan(X).any():
                nan_count = np.isnan(X).sum()
                results["checks"].append(f"⚠️  {nan_count} missing values in features")
            else:
                results["checks"].append("✅ No missing values")
            
            print(f"✅ {dataset_name} validation completed")
            
        except Exception as e:
            results = {
                "status": "❌ FAILED",
                "dataset_name": dataset_name,
                "error": str(e),
                "traceback": traceback.format_exc()
            }
            print(f"❌ {dataset_name} validation failed: {e}")
        
        return results
    
    def validate_multi_dataset_fusion(self) -> Dict[str, Any]:
        """Validate multi-dataset fusion capability."""
        dataset_name = "Multi-Dataset Fusion"
        print(f"\n🔍 Validating {dataset_name}...")
        
        try:
            # Find available datasets
            available_datasets = []
            dataset_dirs = {}
            
            potential_datasets = [
                ("brazilian_rockfall_slope", BrazilianRockfallSlopeLoader),
                ("railway_rockfall", RailwayRockfallLoader)
            ]
            
            for dir_name, loader_class in potential_datasets:
                dataset_dir = self.base_data_dir / dir_name
                if dataset_dir.exists():
                    dataset_dirs[dir_name] = str(dataset_dir)
                    available_datasets.append(dir_name)
            
            if len(available_datasets) < 2:
                results = {
                    "status": "⚠️  SKIPPED",
                    "dataset_name": dataset_name,
                    "reason": f"Need at least 2 datasets for fusion. Available: {available_datasets}",
                    "available_datasets": available_datasets
                }
                print(f"⚠️  {dataset_name} validation skipped: Insufficient datasets")
                return results
            
            # Test multi-dataset loader
            multi_loader = MultiDatasetLoader(dataset_dirs)
            X, y, metadata = multi_loader.load_data()
            
            results = {
                "status": "✅ PASSED",
                "dataset_name": dataset_name,
                "available_datasets": available_datasets,
                "num_samples": len(X),
                "num_features": X.shape[1] if X.ndim > 1 else 1,
                "fusion_method": metadata.get('fusion_method'),
                "dataset_contributions": metadata.get('dataset_contributions', {}),
                "checks": []
            }
            
            # Check fusion consistency
            if X.shape[0] != y.shape[0]:
                results["checks"].append("❌ Fused X and y have different number of samples")
                results["status"] = "❌ FAILED"
            else:
                results["checks"].append("✅ Fused data is consistent")
            
            # Check dataset contributions
            total_samples = sum(metadata.get('dataset_contributions', {}).values())
            if total_samples != len(X):
                results["checks"].append("⚠️  Dataset contribution counts don't match total")
            else:
                results["checks"].append("✅ Dataset contributions are consistent")
            
            for dataset, count in metadata.get('dataset_contributions', {}).items():
                results["checks"].append(f"📊 {dataset}: {count} samples")
            
            print(f"✅ {dataset_name} validation completed")
            
        except Exception as e:
            results = {
                "status": "❌ FAILED",
                "dataset_name": dataset_name,
                "error": str(e),
                "traceback": traceback.format_exc()
            }
            print(f"❌ {dataset_name} validation failed: {e}")
        
        return results
    
    def run_all_validations(self) -> Dict[str, Dict[str, Any]]:
        """Run all dataset validations."""
        print("🚀 Starting comprehensive dataset validation...\n")
        
        validation_functions = [
            self.validate_brazilian_rockfall,
            self.validate_rocknet_seismic,
            self.validate_open_pit_mine,
            self.validate_railway_rockfall,
            self.validate_multi_dataset_fusion
        ]
        
        results = {}
        
        for validation_func in validation_functions:
            try:
                result = validation_func()
                results[result['dataset_name']] = result
            except Exception as e:
                dataset_name = validation_func.__name__.replace('validate_', '').replace('_', ' ').title()
                results[dataset_name] = {
                    "status": "❌ FAILED",
                    "dataset_name": dataset_name,
                    "error": f"Validation function failed: {e}",
                    "traceback": traceback.format_exc()
                }
        
        return results
    
    def generate_validation_report(self, results: Dict[str, Dict[str, Any]]):
        """Generate a comprehensive validation report."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.base_data_dir.parent / f"validation_report_{timestamp}.md"
        
        with open(report_file, 'w') as f:
            f.write("# Dataset Validation Report\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Summary
            passed = sum(1 for r in results.values() if r['status'] == '✅ PASSED')
            failed = sum(1 for r in results.values() if r['status'] == '❌ FAILED')
            skipped = sum(1 for r in results.values() if '⚠️' in r['status'])
            
            f.write("## Summary\n\n")
            f.write(f"- ✅ Passed: {passed}\n")
            f.write(f"- ❌ Failed: {failed}\n")
            f.write(f"- ⚠️  Skipped: {skipped}\n")
            f.write(f"- 📊 Total: {len(results)}\n\n")
            
            # Detailed results
            f.write("## Detailed Results\n\n")
            
            for dataset_name, result in results.items():
                f.write(f"### {dataset_name}\n\n")
                f.write(f"**Status:** {result['status']}\n\n")
                
                if 'error' in result:
                    f.write(f"**Error:** `{result['error']}`\n\n")
                    if 'traceback' in result:
                        f.write("**Traceback:**\n```\n")
                        f.write(result['traceback'])
                        f.write("\n```\n\n")
                else:
                    # Write dataset statistics
                    if 'num_samples' in result:
                        f.write(f"- **Samples:** {result['num_samples']}\n")
                    if 'num_features' in result:
                        f.write(f"- **Features:** {result['num_features']}\n")
                    if 'num_classes' in result:
                        f.write(f"- **Classes:** {result['num_classes']}\n")
                    if 'class_distribution' in result:
                        f.write(f"- **Class Distribution:** {result['class_distribution']}\n")
                    
                    # Write validation checks
                    if 'checks' in result:
                        f.write("\n**Validation Checks:**\n\n")
                        for check in result['checks']:
                            f.write(f"- {check}\n")
                    
                    f.write("\n")
                
                f.write("---\n\n")
        
        print(f"\n📄 Detailed report saved to: {report_file}")
        return report_file
    
    def print_summary(self, results: Dict[str, Dict[str, Any]]):
        """Print validation summary."""
        print("\n" + "="*80)
        print("VALIDATION SUMMARY")
        print("="*80)
        
        for dataset_name, result in results.items():
            status_icon = "✅" if result['status'] == '✅ PASSED' else "❌" if '❌' in result['status'] else "⚠️"
            print(f"{status_icon} {dataset_name}: {result['status']}")
            
            if result['status'] == '✅ PASSED':
                if 'num_samples' in result:
                    print(f"   📊 {result['num_samples']} samples")
                if 'num_features' in result:
                    print(f"   📈 {result['num_features']} features")
        
        print("="*80)


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate specialized rockfall datasets")
    parser.add_argument('--data-dir', default='data/raw', help='Base data directory')
    parser.add_argument('--dataset', choices=[
        'brazilian', 'rocknet', 'openpit', 'railway', 'fusion', 'all'
    ], default='all', help='Specific dataset to validate')
    parser.add_argument('--report', action='store_true', help='Generate detailed report')
    
    args = parser.parse_args()
    
    validator = DatasetValidator(args.data_dir)
    
    if args.dataset == 'all':
        results = validator.run_all_validations()
    else:
        # Map arguments to validation functions
        validation_map = {
            'brazilian': validator.validate_brazilian_rockfall,
            'rocknet': validator.validate_rocknet_seismic,
            'openpit': validator.validate_open_pit_mine,
            'railway': validator.validate_railway_rockfall,
            'fusion': validator.validate_multi_dataset_fusion
        }
        
        if args.dataset in validation_map:
            result = validation_map[args.dataset]()
            results = {result['dataset_name']: result}
        else:
            print(f"Unknown dataset: {args.dataset}")
            return
    
    # Print summary
    validator.print_summary(results)
    
    # Generate report if requested
    if args.report:
        validator.generate_validation_report(results)


if __name__ == "__main__":
    main()