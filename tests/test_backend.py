"""Unit tests for backend — core functionality."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))


class TestConfig:
    def test_config_importable(self):
        from src.config import settings
        assert settings is not None
