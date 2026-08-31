---
sidebar_position: 1
---

# Install Teleopit

Install only the parts you need. All commands below run from the repository
root and require Python 3.10 or newer.

## 1. Get the Code

```bash
git clone https://github.com/BotRunner64/Teleopit.git
cd Teleopit
```

You only need Git submodules for a physical G1 or optional LinkerHand control;
those steps appear later on this page.

## 2. Create a Python Environment

Choose one environment tool. Do not run all three sections.

### uv

```bash
uv venv --python 3.10
source .venv/bin/activate
```

When this page shows `pip install`, you may use `uv pip install` instead.

### pip and venv

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### Conda

```bash
conda create -n teleopit python=3.10
conda activate teleopit
```

Conda creates the environment; use `pip install` inside that environment to
install Teleopit.

## 3. Install the Profile You Need

Each extra includes the base Teleopit package. Start with the row matching your
goal; you can install another extra later in the same environment.

| Goal | Install command | What it adds |
|------|-----------------|--------------|
| Run a motion controller in MuJoCo | `pip install -e .` | Core inference, GMR, MuJoCo and ONNX Runtime |
| Use Pico in simulation or on G1 | `pip install -e '.[pico4]'` | Pico receiver plus the sim2real runtime |
| Replay BVH on a physical G1 without Pico | `pip install -e '.[sim2real]'` | G1 runtime and OpenCV |
| Train a controller | `pip install -e '.[train]'` | mjlab, RSL-RL and experiment loggers |
| Record Pico sim2real episodes | `pip install -e '.[recording]'` | Pico runtime and MP4 writing |
| Review saved recordings | `pip install -e '.[review]'` | OpenCV and the MuJoCo/Viser reviewer |
| Use OpenNeck with Pico | `pip install -e '.[openneck]'` | Pico runtime and the OpenNeck driver |
| Run the test suite | `pip install -e '.[dev]'` | pytest and coverage tools |

## 4. Download the Matching Assets

The Python package does not contain robot meshes, policies or motion datasets.
Install the default ModelScope downloader once:

```bash
pip install modelscope
```

Then download the bundle for your goal:

| Goal | Command |
|------|---------|
| Simulation, Pico VR or G1 inference | `python scripts/setup/download_assets.py --only robots gmr ckpt bvh` |
| Training from the distributed datasets | `python scripts/setup/download_assets.py --only robots data` |
| Everything | `python scripts/setup/download_assets.py` |

Use HuggingFace instead of ModelScope when needed:

```bash
python scripts/setup/download_assets.py \
    --source huggingface \
    --only robots gmr ckpt bvh
```

The inference bundle creates the `track_g1` and `track_g1_neck_o6` ONNX/checkpoint
pairs under `ckpt/`, plus the G1 model files, GMR files and a sample BVH under
their expected project paths. See
[Assets](../reference/resources/assets) for the complete inventory
and asset group mapping.

## 5. Additional Setup for a Physical G1

Build the C++ DDS bridge on the computer that will run Teleopit:

```bash
git submodule update --init --recursive
bash scripts/setup/setup_g1_bridge.sh
```

The bridge is required for both Pico and BVH control on a real G1. See
[Companion Projects](../reference/companion-projects#g1-bridge-sdk) if the
build or robot connection fails.

## 6. Optional Hardware

### LinkerHand L6 or O6

Only install these local packages when `hands.enabled=true`:

```bash
git submodule update --init --recursive
pip install -e third_party/linkerhand-python-sdk
pip install -e third_party/somehand
bash scripts/setup/download_somehand_assets.sh
```

### OpenNeck

The `openneck` extra already includes the Pico profile. Calibrate the device
before enabling it:

```bash
pip install -e '.[openneck]'
openneck calibrate
```

Teleopit uses the OpenNeck angle API. Old normalized calibration fields are not
supported.

### RealSense Recording or Preview

Install `pyrealsense2` separately when a RealSense camera is enabled. On Arm
machines, use conda-forge:

```bash
conda install -c conda-forge pyrealsense2
```

Pico body tracking itself does not require RealSense.

## 7. Verify the Environment

Run the core import check:

```bash
python -c "import teleopit; print('teleopit OK')"
```

If you installed Pico or training dependencies, run the matching check:

```bash
python -c "from pico_bridge import PicoBridge; print('Pico OK')"
python -c "import train_mimic.tasks; print('training OK')"
```

For an inference profile with the `robots gmr ckpt bvh` assets, finish with one
sample simulation:

```bash
python scripts/run/run_sim.py \
    controller.policy_path=ckpt/track_g1.onnx \
    input.bvh_file=data/sample_bvh/aiming1_subject1.bvh
```

The installation is ready when a MuJoCo window opens and the simulated G1
follows the sample motion. Close the window to stop, then continue with one of
the four task-based tutorials.
