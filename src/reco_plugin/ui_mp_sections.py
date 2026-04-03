from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QLabel, 
    QLineEdit, 
    QVBoxLayout, 
    QHBoxLayout, 
    QComboBox, 
    QCheckBox,
    QSpinBox,
    QPushButton,
    QGroupBox,
    QWidget,
)

from ._link import *
from .ui_sections import (
    add_energy_layout,
    add_pixel_size_layout,
    add_distance_object_detector_layout,
    add_unsharp_mask_layout,
)

def add_general_parameters_section(widget):
    """
    Add multipaganin section to the widget.
    """
    group_box = QGroupBox("General Parameters")
    layout = QVBoxLayout()

    widget.variables_layout = QVBoxLayout()  # Initialize variables_layout
    layout.addLayout(widget.variables_layout)

    add_energy_layout(widget)
    add_pixel_size_layout(widget)
    add_distance_object_detector_layout(widget)
    add_unsharp_mask_layout(widget)
    add_acquisition_type_layout(widget)
    add_center_of_rotation_layout(widget)

    group_box.setLayout(layout)
    widget.layout().addWidget(group_box)

def add_multi_paganin_sections(widget):
    """
    Add multipaganin section to the widget.
    """
    widget.paganin_sections = []

    # Conteneur pour les sections dynamiques
    widget.sections_container = QWidget()
    widget.sections_layout = QVBoxLayout()
    widget.sections_layout.setAlignment(Qt.AlignTop)
    widget.sections_container.setLayout(widget.sections_layout)
    widget.layout().addWidget(widget.sections_container)

    # Layout for buttons
    button_layout = QHBoxLayout()

    # Bouton pour ajouter une nouvelle section
    widget.add_section_button = QPushButton("+")
    widget.add_section_button.clicked.connect(lambda: add_mpaganin_section(widget))
    button_layout.addWidget(widget.add_section_button)

    # Bouton pour supprimer la dernière section
    widget.remove_section_button = QPushButton("-")
    widget.remove_section_button.clicked.connect(lambda: remove_last_paganin_section(widget))
    button_layout.addWidget(widget.remove_section_button)

    widget.layout().addLayout(button_layout)

def remove_last_paganin_section(widget):
    """
    Remove the last added Paganin section from the widget.
    """
    if widget.paganin_sections:
        last_section = widget.paganin_sections.pop()  # Remove from the list
        widget.sections_layout.removeWidget(last_section)  # Remove from the layout
        last_section.deleteLater()  # Delete the widget

def add_mpaganin_section(widget):
    index = len(widget.paganin_sections)
    section = create_paganin_section(widget, index)

    widget.sections_layout.addWidget(section)  # Append the section at the bottom
    widget.paganin_sections.append(section)  # Add to the end of the list

def create_paganin_section(widget, index: int) -> QGroupBox:
    group = QGroupBox(f"Paganin {index}")
    layout = QVBoxLayout()

    # Tu peux remplacer ce QLabel par de vrais champs (QLineEdit, QDoubleSpinBox, etc.)
    layout = QVBoxLayout()

    db_layout = QHBoxLayout()
    db_label = QLabel("Delta Beta Ratio:")
    db_input = QLineEdit()
    db_layout.addWidget(db_label)
    db_layout.addWidget(db_input)
    
    threshold_layout = QHBoxLayout()
    threshold_label = QLabel("Threshold:")
    threshold_input = QLineEdit()
    threshold_layout.addWidget(threshold_label)
    threshold_layout.addWidget(threshold_input)

    test_button = QPushButton("Test")
    test_button.clicked.connect(lambda: print(f"Test button clicked for section {index}"))

    layout.addLayout(db_layout)
    layout.addLayout(threshold_layout)
    layout.addWidget(test_button)

    group.setLayout(layout)
    return group

def add_acquisition_type_layout(widget):
    layout = QHBoxLayout()
    layout.addWidget(QLabel("Acquisition Type:"))
    widget.acquisition_type_selection = QComboBox()
    widget.acquisition_type_selection.addItems(["Standard Acquisition", "Half Acquisition"])
    widget.acquisition_type_selection.setCurrentIndex(0)  # Set default to Standard Acquisition
    layout.addWidget(widget.acquisition_type_selection)
    widget.variables_layout.addLayout(layout)

def add_center_of_rotation_layout(widget):
    layout = QHBoxLayout()
    layout.addWidget(QLabel("Center of Rotation:"))
    widget.center_of_rotation_input = QLineEdit()
    widget.center_of_rotation_input.setText(
        str(widget.experiment.center_of_rotation) if widget.experiment.center_of_rotation is not None else ""
    )
    layout.addWidget(widget.center_of_rotation_input)
    widget.variables_layout.addLayout(layout)