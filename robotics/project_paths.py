"""Shared paths for the robotics scripts.

The OT-2 digital twin (sim_class.py, URDF files, meshes and textures) is not part
of this repository. See simulation/README.md for where to put it.
"""
import os
import sys

ROBOTICS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(ROBOTICS_DIR)
SIM_DIR = os.path.join(REPO_ROOT, "simulation")
MODELS_DIR = os.path.join(REPO_ROOT, "models")
OUTPUT_DIR = os.path.join(REPO_ROOT, "outputs")
CV_DATA_PATH = os.path.join(REPO_ROOT, "data", "root_tip_coordinates.json")


def enter_simulation_dir():
    """Make sim_class importable and switch to the simulation folder.

    sim_class.py loads its URDF files and textures with paths relative to the
    working directory, so the scripts have to run from inside that folder.
    """
    if not os.path.isfile(os.path.join(SIM_DIR, "sim_class.py")):
        sys.exit(f"OT-2 simulation not found in {SIM_DIR}. See simulation/README.md.")
    os.chdir(SIM_DIR)
    if SIM_DIR not in sys.path:
        sys.path.insert(0, SIM_DIR)
