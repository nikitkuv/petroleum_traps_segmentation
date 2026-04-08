"""
Test suite for Geology Traps Segmentation project.

This module contains comprehensive tests for:
1. Input data validation (PNG and CPS formats)
2. Dataset loading (GeologyTrapsDataset)
3. Training readiness checks
4. Training process verification

Usage:
    # Run all tests
    pytest tests/ -v
    
    # Run specific test categories
    pytest tests/test_data_validation.py -v
    pytest tests/test_dataset.py -v
    pytest tests/test_training.py -v
    
    # Run with coverage
    pytest tests/ --cov=. --cov-report=html
    
    # Run quick smoke tests
    pytest tests/ -m "smoke" -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
