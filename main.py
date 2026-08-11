"""Run the full training + evaluation pipeline (kept for compatibility).

Equivalent to: python train_model.py
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from train_model import run_training_pipeline

if __name__ == "__main__":
    run_training_pipeline()
