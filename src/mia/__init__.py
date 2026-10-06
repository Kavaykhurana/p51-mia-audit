"""P51 Membership-Inference Audit: shared library for the notebooks, the Streamlit app and the tests."""
__version__ = "1.0.0"

from .config import Experiment, ModelConfig, load_experiment
from .data import CLASS_NAMES

__all__ = ["CLASS_NAMES", "Experiment", "ModelConfig", "load_experiment", "__version__"]
