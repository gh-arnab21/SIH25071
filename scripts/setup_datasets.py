#!/usr/bin/env python3
"""
Dataset setup and validation script for specialized rockfall datasets.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, Any, List
import yaml
import requests
from urllib.parse import urlparse

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.specialized_loaders import (
    BrazilianRockfallSlopeLoader,
    RockNetSeismicLoader, 
    OpenPitMineDetectionLoader,
    RailwayRockfallLoader
)


class DatasetSetupManager:
    """Manages setup and validation of specialized datasets."""
    
    def __init__(self, base_data_dir: str = "data/raw"):
        self.base_data_dir = Path(base_data_dir)
        self.base_data_dir.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Dataset configuration
        self.datasets_config = {
            'brazilian_rockfall_slope': {
                'dir': 'brazilian_rockfall_slope',
                'description': 'Brazilian Rockfall Slope Dataset (220 samples, CSV)',
                'expected_files': ['slope_data.csv'],
                'loader_class': BrazilianRockfallSlopeLoader,
                'download_info': {
                    'source': 'Research paper supplementary material',
                    'manual': True,
                    'instructions': 'Download CSV file from paper supplementary material'
                }
            },
            'rocknet_seismic': {
                'dir': 'rocknet_seismic',
                'description': 'RockNet Seismic Dataset (Taiwan rockfall/earthquake)',
                'expected_dirs': ['data/earthquake', 'data/rockfall', 'data/noise'],
                'loader_class': RockNetSeismicLoader,
                'download_info': {
                    'source': 'https://github.com/tso1257771/RockNet',
                    'manual': False,
                    'git_repo': 'https://github.com/tso1257771/RockNet.git'
                }
            },
            'open_pit_mine_detection': {
                'dir': 'open_pit_mine_detection',
                'description': 'Open Pit Mine Object Detection Dataset (1.06 GB)',
                'expected_dirs': ['images', 'annotations'],
                'loader_class': OpenPitMineDetectionLoader,
                'download_info': {
                    'source': 'https://figshare.com/articles/dataset/Open_Pit_Mine_Object_Detection_Dataset/27300960',
                    'manual': True,
                    'instructions': 'Download and extract from Figshare'
                }
            },
            'railway_rockfall': {
                'dir': 'railway_rockfall',
                'description': 'Railway Rockfall Detection Dataset (1,733 samples)',
                'expected_files': ['features.csv'],
                'loader_class': RailwayRockfallLoader,
                'download_info': {
                    'source': 'Research paper or contact authors',
                    'manual': True,
                    'instructions': 'Contact paper authors or check supplementary material'
                }
            }
        }
    
    def check_dataset_status(self) -> Dict[str, Dict[str, Any]]:
        """Check the status of all datasets."""
        status = {}
        
        for dataset_name, config in self.datasets_config.items():
            dataset_dir = self.base_data_dir / config['dir']
            dataset_status = {
                'exists': dataset_dir.exists(),
                'path': str(dataset_dir),
                'files_present': False,
                'can_load': False,
                'description': config['description']
            }
            
            if dataset_status['exists']:
                # Check for expected files/directories
                if 'expected_files' in config:
                    files_exist = all((dataset_dir / f).exists() for f in config['expected_files'])
                    dataset_status['files_present'] = files_exist
                elif 'expected_dirs' in config:
                    dirs_exist = all((dataset_dir / d).exists() for d in config['expected_dirs'])
                    dataset_status['files_present'] = dirs_exist
                
                # Try to load with the data loader
                if dataset_status['files_present']:
                    try:
                        loader = config['loader_class'](str(dataset_dir))
                        X, y, metadata = loader.load_data()
                        dataset_status['can_load'] = True
                        dataset_status['num_samples'] = metadata.get('num_samples', len(X))
                        dataset_status['num_features'] = metadata.get('num_features', X.shape[1] if X.ndim > 1 else 1)
                    except Exception as e:
                        dataset_status['load_error'] = str(e)
            
            status[dataset_name] = dataset_status
        
        return status
    
    def print_status_report(self):
        """Print a comprehensive status report."""
        status = self.check_dataset_status()
        
        print("="*80)
        print("DATASET STATUS REPORT")
        print("="*80)
        
        for dataset_name, info in status.items():
            print(f"\n📊 {dataset_name.upper()}")
            print(f"   Description: {info['description']}")
            print(f"   Path: {info['path']}")
            print(f"   Directory exists: {'✅' if info['exists'] else '❌'}")
            print(f"   Files present: {'✅' if info['files_present'] else '❌'}")
            print(f"   Can load: {'✅' if info['can_load'] else '❌'}")
            
            if info['can_load']:
                print(f"   📈 Samples: {info.get('num_samples', 'N/A')}")
                print(f"   📊 Features: {info.get('num_features', 'N/A')}")
            elif 'load_error' in info:
                print(f"   ❌ Error: {info['load_error']}")
    
    def create_dataset_directories(self):
        """Create directory structure for all datasets."""
        print("Creating dataset directories...")
        
        for dataset_name, config in self.datasets_config.items():
            dataset_dir = self.base_data_dir / config['dir']
            dataset_dir.mkdir(parents=True, exist_ok=True)
            print(f"✅ Created: {dataset_dir}")
    
    def download_available_datasets(self):
        """Download datasets that can be automatically downloaded."""
        print("\nDownloading available datasets...")
        
        for dataset_name, config in self.datasets_config.items():
            download_info = config['download_info']
            
            if not download_info['manual']:
                dataset_dir = self.base_data_dir / config['dir']
                
                if 'git_repo' in download_info:
                    self._clone_git_repository(download_info['git_repo'], dataset_dir)
    
    def _clone_git_repository(self, repo_url: str, target_dir: Path):
        """Clone a git repository."""
        try:
            import subprocess
            
            if target_dir.exists() and any(target_dir.iterdir()):
                print(f"⏭️  Directory already exists: {target_dir}")
                return
            
            print(f"📥 Cloning {repo_url} to {target_dir}")
            result = subprocess.run(['git', 'clone', repo_url, str(target_dir)], 
                                 capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"✅ Successfully cloned to {target_dir}")
            else:
                print(f"❌ Failed to clone: {result.stderr}")
                
        except FileNotFoundError:
            print("❌ Git not found. Please install git or download manually.")
        except Exception as e:
            print(f"❌ Error cloning repository: {e}")
    
    def generate_download_instructions(self):
        """Generate download instructions for manual datasets."""
        instructions_file = self.base_data_dir / "download_instructions.md"
        
        with open(instructions_file, 'w') as f:
            f.write("# Dataset Download Instructions\n\n")
            f.write("This file contains instructions for downloading datasets that require manual download.\n\n")
            
            for dataset_name, config in self.datasets_config.items():
                download_info = config['download_info']
                
                if download_info['manual']:
                    f.write(f"## {dataset_name.replace('_', ' ').title()}\n\n")
                    f.write(f"**Description:** {config['description']}\n\n")
                    f.write(f"**Source:** {download_info['source']}\n\n")
                    f.write(f"**Instructions:** {download_info['instructions']}\n\n")
                    f.write(f"**Target Directory:** `{self.base_data_dir / config['dir']}`\n\n")
                    
                    if 'expected_files' in config:
                        f.write("**Expected Files:**\n")
                        for file in config['expected_files']:
                            f.write(f"- {file}\n")
                    elif 'expected_dirs' in config:
                        f.write("**Expected Directories:**\n")
                        for dir in config['expected_dirs']:
                            f.write(f"- {dir}/\n")
                    
                    f.write("\n---\n\n")
        
        print(f"📝 Download instructions saved to: {instructions_file}")
    
    def setup_all(self):
        """Run complete setup process."""
        print("🚀 Starting dataset setup process...\n")
        
        # Create directories
        self.create_dataset_directories()
        
        # Download available datasets
        self.download_available_datasets()
        
        # Generate instructions for manual downloads
        self.generate_download_instructions()
        
        # Show status report
        print("\n" + "="*80)
        self.print_status_report()
        
        print("\n" + "="*80)
        print("SETUP COMPLETE!")
        print("="*80)
        print("Next steps:")
        print("1. Download manual datasets following instructions in data/raw/download_instructions.md")
        print("2. Run validation: python scripts/validate_datasets.py")
        print("3. Start training: python scripts/train_model.py --dataset-type <dataset_name>")


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Setup specialized rockfall datasets")
    parser.add_argument('--data-dir', default='data/raw', help='Base data directory')
    parser.add_argument('--action', choices=['setup', 'status', 'download', 'instructions'], 
                       default='setup', help='Action to perform')
    
    args = parser.parse_args()
    
    manager = DatasetSetupManager(args.data_dir)
    
    if args.action == 'setup':
        manager.setup_all()
    elif args.action == 'status':
        manager.print_status_report()
    elif args.action == 'download':
        manager.download_available_datasets()
    elif args.action == 'instructions':
        manager.generate_download_instructions()


if __name__ == "__main__":
    main()