import pytest
import numpy as np
import torch
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "smoke: quick smoke tests for basic functionality"
    )
    config.addinivalue_line(
        "markers", "slow: tests that take longer to run (e.g., full training)"
    )
    config.addinivalue_line(
        "markers", "integration: integration tests requiring multiple components"
    )


@pytest.fixture(autouse=True)
def set_cpu_device(monkeypatch):
    """Force CPU usage for all tests to ensure reproducibility."""
    monkeypatch.setattr('settings.settings.DEVICE', 'cpu')


@pytest.fixture
def sample_tensor():
    """Create a sample tensor for testing."""
    return torch.randn(2, 3, 64, 64)


@pytest.fixture
def sample_binary_mask():
    """Create a sample binary mask."""
    mask = torch.zeros(2, 1, 64, 64)
    mask[:, :, 20:40, 20:40] = 1.0
    return mask


@pytest.fixture
def sample_rgb_image():
    """Create a sample RGB image as numpy array."""
    return np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)


@pytest.fixture
def sample_grayscale_image():
    """Create a sample grayscale image as numpy array."""
    return np.random.randint(0, 255, (64, 64), dtype=np.uint8)
