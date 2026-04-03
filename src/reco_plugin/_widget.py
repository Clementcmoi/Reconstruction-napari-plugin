from typing import TYPE_CHECKING
from qtpy.QtWidgets import (
    QLabel, 
    QVBoxLayout, 
    QWidget,
    QSpacerItem, 
    QSizePolicy,
)

from .ui_sections import *
from .ui_mp_sections import *
from .utils.layer_utils import LayerUtils
from .utils.experiment import Experiment, mpExperiment

if TYPE_CHECKING:
    import napari

class ReconstructionWidget(QWidget):
    def __init__(self, napari_viewer: 'napari.Viewer'):
        super().__init__()
        self.viewer = napari_viewer
        self.experiment = Experiment()
        self.setup_ui()
        LayerUtils.connect_signals(self) 

    def setup_ui(self):
        self.setLayout(QVBoxLayout())
        self.layout().addWidget(QLabel("Reconstruction"))

        add_sample_selection_section(self)
        add_preprocessing_section(self)
        add_paganin_section(self)
        add_double_flatfield_section(self)
        add_angles_section(self)
        add_center_of_rotation_section(self)
        add_reconstruction_section(self)
        add_process_section(self)

        LayerUtils.update_layer_selections(self)
        self.layout().addSpacerItem(
            QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)
        )

class MultiPaganinWidget(QWidget):
    def __init__(self, napari_viewer: 'napari.Viewer'):
        super().__init__()
        self.viewer = napari_viewer
        self.experiment = mpExperiment()
        self.setup_ui()
        LayerUtils.connect_signals(self) 

    def setup_ui(self):
        self.setLayout(QVBoxLayout())
        self.layout().addWidget(QLabel("Multi-Paganin"))

        add_sample_selection_section(self)
        add_preprocessing_section(self)
        add_general_parameters_section(self)
        add_multi_paganin_sections(self)
        
        LayerUtils.update_layer_selections(self)
        self.layout().addSpacerItem(
            QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)
        )