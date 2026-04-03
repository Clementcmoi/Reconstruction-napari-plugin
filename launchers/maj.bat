@echo off
REM Path to your Miniconda or Anaconda installation (edit as needed)
CALL <path_to_conda>\Scripts\activate.bat

REM Activate your napari environment (edit as needed)
CALL conda activate <your_napari_env>

REM Go to your plugin folder (edit as needed)
cd /d "<path_to_your_plugin>"

REM Update the plugin from Git
git pull origin main

REM (Optional) Return to your preferred location
cd /d "<your_home_directory>"

REM Launch Napari
napari