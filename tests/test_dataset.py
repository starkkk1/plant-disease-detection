import os
import unittest
import tempfile
import json
import shutil
from PIL import Image
import torch
import yaml

from src.utils.config import load_config
from src.data.dataset import TomatoDataset, get_dataloaders
from scripts.prepare_dataset import prepare_dataset, TARGET_CLASSES

class TestDatasetPipeline(unittest.TestCase):
    def setUp(self):
        # Create temporary directories for testing
        self.test_dir = tempfile.mkdtemp()
        self.raw_dir = os.path.join(self.test_dir, "raw")
        self.processed_dir = os.path.join(self.test_dir, "processed")
        self.reports_dir = os.path.join(self.test_dir, "reports")
        self.logs_dir = os.path.join(self.test_dir, "logs")
        
        os.makedirs(self.raw_dir)
        os.makedirs(self.processed_dir)
        os.makedirs(self.reports_dir)
        os.makedirs(self.logs_dir)
        
        # Create dummy configuration file
        self.config_path = os.path.join(self.test_dir, "config.yaml")
        self.class_map_path = os.path.join(self.processed_dir, "class_to_idx.json")
        
        self.config_data = {
            'project': {
                'name': 'test-project',
                'seed': 42
            },
            'data': {
                'raw_root': self.raw_dir,
                'processed_root': self.processed_dir,
                'train_dir': os.path.join(self.processed_dir, "train"),
                'val_dir': os.path.join(self.processed_dir, "val"),
                'test_dir': os.path.join(self.processed_dir, "test"),
                'class_map': self.class_map_path,
                'image_size': 224,
                'batch_size': 2,
                'num_workers': 0
            },
            'augmentation': {
                'train': {
                    'random_horizontal_flip': True,
                    'rotation_degrees': 15,
                    'color_jitter_brightness': 0.2,
                    'color_jitter_contrast': 0.2,
                    'random_resized_crop': True,
                    'crop_scale_min': 0.8,
                    'crop_scale_max': 1.0
                },
                'val_test': {
                    'resize': 256,
                    'center_crop': 224
                }
            },
            'normalization': {
                'mean': [0.485, 0.456, 0.406],
                'std': [0.229, 0.224, 0.225]
            },
            'output': {
                'reports_dir': self.reports_dir,
                'log_dir': self.logs_dir
            }
        }
        
        with open(self.config_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.config_data, f)
            
        # Create some dummy raw images in target class folders
        # We will create folders that match both tomato and non-tomato classes to test filtering
        self.classes_to_create = TARGET_CLASSES + ["Apple___healthy", "Corn___Common_rust"]
        
        for cls in self.classes_to_create:
            class_raw_dir = os.path.join(self.raw_dir, cls)
            os.makedirs(class_raw_dir)
            
            # Create 10 dummy images per class to test splitting
            for i in range(10):
                img = Image.new("RGB", (100, 100), color=(i * 20, 100, 100))
                img.save(os.path.join(class_raw_dir, f"img_{i}.jpg"))
                
    def tearDown(self):
        import logging
        logging.shutdown()
        shutil.rmtree(self.test_dir)
        
    def test_prepare_dataset_force_split(self):
        # Run prepare dataset with force_split=True to trigger splitting logic
        prepare_dataset(self.config_path, filter_tomato=True, force_split=True, run_stats=True)
        
        # Verify processed subdirectories exist
        splits = ['train', 'val', 'test']
        for split in splits:
            split_dir = os.path.join(self.processed_dir, split)
            self.assertTrue(os.path.exists(split_dir))
            
            # Check that only target tomato classes exist
            processed_classes = os.listdir(split_dir)
            self.assertEqual(len(processed_classes), len(TARGET_CLASSES))
            for cls in processed_classes:
                self.assertIn(cls, TARGET_CLASSES)
                self.assertNotIn(cls, ["Apple___healthy", "Corn___Common_rust"])
                
        # Check that class_to_idx.json is created
        self.assertTrue(os.path.exists(self.class_map_path))
        with open(self.class_map_path, 'r', encoding='utf-8') as f:
            class_to_idx = json.load(f)
        self.assertEqual(len(class_to_idx), len(TARGET_CLASSES))
        self.assertIn("Tomato___healthy", class_to_idx)
        
        # Check that reports/dataset_summary.md is created
        summary_path = os.path.join(self.reports_dir, 'dataset_summary.md')
        self.assertTrue(os.path.exists(summary_path))
        
        # Verify dataloaders can load batches
        train_loader, val_loader, test_loader = get_dataloaders(self.config_data)
        
        self.assertEqual(len(train_loader.dataset), 70) # 10 classes * 7 images for train
        self.assertEqual(len(val_loader.dataset), 10)   # 10 classes * 1 image for val
        self.assertEqual(len(test_loader.dataset), 20)  # 10 classes * 2 images for test (10 - 7 - 1 = 2)
        
        # Load a batch from train
        images, labels = next(iter(train_loader))
        self.assertEqual(images.shape, (2, 3, 224, 224)) # batch_size=2
        self.assertEqual(labels.shape, (2,))

if __name__ == "__main__":
    unittest.main()
