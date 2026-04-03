# Reconstruction Napari Plugin

This project is a Napari plugin for GPU-accelerated tomographic reconstruction, including Paganin phase retrieval.

## Installation

### Option 1 — Install from Napari Hub (recommended)

Once published, the plugin can be installed directly from the Napari plugin manager:

1. Open **Napari**
2. Go to **Plugins > Install/Uninstall Plugins**
3. Search for **"napari-reco-plugin"**
4. Click **Install**

Alternatively, install via pip:

```bash
pip install napari-reco-plugin
```

---

### Option 2 — Manual installation (development)

#### 1. Create and activate a conda environment

```bash
conda create -n napari-env python=3.12
conda activate napari-env
```

#### 2. Install main dependencies

```bash
conda install -c conda-forge napari pyqt
conda install -c astra-toolbox -c nvidia astra-toolbox

# Ensure compatibility with ASTRA
pip install numpy==1.26.4

# Install CuPy depending on your CUDA version
pip install cupy-cuda12x  # or cupy-cuda11x
```

#### 3. Clone the repository

```bash
git clone https://github.com/Clementcmoi/Reconstruction.git
cd Reconstruction
```

#### 4. Install the plugin

```bash
pip install -e .
```

---

## Requirements

### Core

* Python ≥ 3.9
* Napari

### Optional (for GPU acceleration)

* CUDA Toolkit
* CuPy
* ASTRA Toolbox

> ⚠️ GPU dependencies are not installed automatically. You must install them manually according to your system configuration.

---

## Usage

Start Napari:

```bash
napari
```

Then open the plugin via:

```
Plugins > Reconstruction Plugin
```

Available tools:

* **Reconstruction**
* **Multi Paganin**

---

## Launchers Templates

Template scripts for launching Napari with the correct environment are available in the `launchers/` folder.

### Usage

1. Copy a `.bat` file from `launchers/`
2. Edit:

   * `<path_to_conda>` → your Anaconda/Miniconda path
   * `<your_napari_env>` → your environment name
3. Double-click to launch

### Desktop shortcut (optional)

* Right-click `.bat` → **Send to Desktop**
* (Optional) Change icon via shortcut properties

---

## Project Structure

* Plugin source code: `src/reco_plugin/`
* Plugin manifest: `src/reco_plugin/napari.yaml`

---

## License

This project is licensed under the MIT License.
