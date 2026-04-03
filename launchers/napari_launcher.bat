@echo off
REM Path to your Miniconda or Anaconda installation (edit as needed)
CALL <path_to_conda>\Scripts\activate.bat

REM Activate your napari environment (edit as needed)
CALL conda activate <your_napari_env>

REM Launch napari
napari