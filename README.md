# Reconstruction Napari Plugin

This project is a Napari plugin for data reconstruction.

## Requirements

- [Python 3.12](https://www.python.org/downloads/release/python-3129/)
- [Anaconda](https://www.anaconda.com/)
- [Napari](https://napari.org/)
- [Cuda Toolkit](https://developer.nvidia.com/cuda-toolkit)
- [Astra Toolbox](https://www.astra-toolbox.com/)
- [Cupy](https://cupy.dev/)
- Other dependencies listed in `requirements.txt` or `pyproject.toml`

## Installation

### 1. Create and activate a conda environment

```bash
conda create -n napari-env python=3.12
conda activate napari-env
```

### 2. Install main dependencies with conda/pip

```bash
conda install -c conda-forge napari pyqt
conda install -c astra-toolbox -c nvidia astra-toolbox
pip install numpy==1.26.4
# Use cupy-cuda12x or cupy-cuda11x depending on your CUDA version
pip install cupy-cuda12x 
```

### 3. Clone the repository

```bash
git clone https://github.com/Clementcmoi/Reconstruction.git
cd Reconstruction
```

### 4. Install requirements

```bash
pip install -r requirements.txt
```

### 5. Install the plugin in editable/development mode

```bash
pip install -e .
```

## Usage

Start Napari:

```bash
napari
```

Then, access the plugin via the Napari plugins menu (`Plugins > Reconstruction Plugin > Reconstruction` or `Multi Paganin`).

## Launchers Templates

Template scripts for launching Napari with the correct environment are available in the `launchers/` folder.  
To use them:

1. **Copy the desired `.bat` file from `launchers/` to your preferred location.**
2. **Edit the file:**  
   - Replace `<path_to_conda>` with the path to your Miniconda/Anaconda installation.
   - Replace `<your_napari_env>` with the name of your conda environment (e.g., `napari-env`).
   - Optionally, adjust the working directory or other paths as needed.

3. **Double-click the `.bat` file to launch Napari with the appropriate environment.**

These templates are not updated automatically by Git and are intended for local customization.

### Create a Desktop Shortcut for a Launcher

To create a desktop shortcut for a launcher:

1. **Right-click** the desired `.bat` file.
2. Select **"Send to" > "Desktop (create shortcut)"**.
3. On your desktop, you can **rename the shortcut** as you wish.

#### Set a Custom Icon for the Shortcut

1. **Right-click** the shortcut on your desktop and select **"Properties"**.
2. Go to the **"Shortcut"** tab and click **"Change Icon..."**.
3. Browse and select the `.ico` file available in the `launchers/` folder (e.g., `launchers/napari.ico`).
   - You can use this Napari logo or any custom icon.
   - If you have a `.png` or other image, convert it to `.ico` using an online converter.
4. Click **OK** to apply the icon.

## Project Structure

- Plugin source code: `src/reco_plugin/`
- Plugin manifest: `src/reco_plugin/napari.yaml`
- Data files: `.npy`, `.tif` (ignored by Git)
- Notebooks: `.ipynb` (ignored by Git)

## License

This project is licensed under the MIT License.