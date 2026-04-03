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
)

from ._link import *

def update_slice_range(widget):
    """
    Adjusts the range of the `slice_selection` according to the selected layer.
    """
    selected_layer_name = widget.sample_selection.currentText()
    if (selected_layer_name in widget.viewer.layers):
        selected_layer = widget.viewer.layers[selected_layer_name]
        if hasattr(selected_layer, 'data'):
            max_slices = selected_layer.data.shape[0] - 1
            widget.slice_selection.setMaximum(max_slices)

def toggle_bigdata(widget, checked):
    """
    Show or hide the batch size row depending on the Big Data checkbox state.
    """
    visible = (checked == Qt.Checked)
    widget.batch_size_label.setVisible(visible)
    widget.batch_size_input.setVisible(visible)


def add_sample_selection_section(widget):
    """
    Add sample and slice selection section to the widget.
    """
    group_box = QGroupBox("Sample Selection")
    layout = QVBoxLayout()

    layout_sample = QHBoxLayout()
    sample_label = QLabel("Sample:")
    widget.sample_selection = QComboBox()
    layout_sample.addWidget(sample_label)
    layout_sample.addWidget(widget.sample_selection)

    layout_slice = QHBoxLayout()
    slice_label = QLabel("Slice:")
    widget.slice_selection = QSpinBox()
    widget.slice_selection.setMinimum(0)
    widget.slice_selection.setMaximum(1000)

    saved_slice_idx = int(widget.experiment.slice_idx) if widget.experiment.slice_idx is not None else 0
    widget.slice_selection.setValue(saved_slice_idx)

    layout_slice.addWidget(slice_label)
    layout_slice.addWidget(widget.slice_selection)

    # Big data checkbox + batch size (hidden by default)
    layout_bigdata = QHBoxLayout()
    widget.bigdata_checkbox = QCheckBox("Big Data Processing")
    widget.batch_size_label = QLabel("Batch size (slices):")
    widget.batch_size_input = QSpinBox()
    widget.batch_size_input.setMinimum(1)
    widget.batch_size_input.setMaximum(1000)
    saved_batch = int(widget.experiment.batch_size) if widget.experiment.batch_size is not None else 50
    widget.batch_size_input.setValue(saved_batch)
    widget.batch_size_label.setVisible(False)
    widget.batch_size_input.setVisible(False)
    widget.bigdata_checkbox.stateChanged.connect(lambda state: toggle_bigdata(widget, state))
    layout_bigdata.addWidget(widget.bigdata_checkbox)
    layout_bigdata.addWidget(widget.batch_size_label)
    layout_bigdata.addWidget(widget.batch_size_input)

    layout.addLayout(layout_sample)
    layout.addLayout(layout_slice)
    layout.addLayout(layout_bigdata)

    group_box.setLayout(layout)

    widget.sample_selection.currentIndexChanged.connect(lambda: update_slice_range(widget))

    widget.layout().addWidget(group_box)

def toggle_field_widgets(widget, checked, layout, label_attr, selection_attr, label_text):
    if checked == Qt.Checked:
        if not getattr(widget, label_attr):
            setattr(widget, label_attr, QLabel(label_text))
        if not getattr(widget, selection_attr):
            combobox = QComboBox()
            combobox.addItems([layer.name for layer in widget.viewer.layers])
            setattr(widget, selection_attr, combobox)
        
        horizontal_layout = QHBoxLayout()
        horizontal_layout.addWidget(getattr(widget, label_attr))
        horizontal_layout.addWidget(getattr(widget, selection_attr))
        layout.addLayout(horizontal_layout)
    else:
        widget_label = getattr(widget, label_attr)
        widget_selection = getattr(widget, selection_attr)
        if widget_label:
            layout.removeWidget(widget_label)
            widget_label.deleteLater()
            setattr(widget, label_attr, None)
        if widget_selection and widget_selection.isVisible():
            layout.removeWidget(widget_selection)
            widget_selection.deleteLater()
            setattr(widget, selection_attr, None)
    widget.layout().update()

def add_darkfield_section(widget):
    """
    Create and return the darkfield-related UI components.
    """
    darkfield_layout = QVBoxLayout()

    checkbox_layout = QHBoxLayout()
    checkbox_layout.addWidget(QLabel("Darkfield:"))
    widget.darkfield_checkbox = QCheckBox()
    widget.darkfield_checkbox.stateChanged.connect(
        lambda state: toggle_field_widgets(widget, state, darkfield_layout, 'darkfield_label', 'darkfield_selection', 'Select darkfield:')
    )
    checkbox_layout.addWidget(widget.darkfield_checkbox)

    darkfield_layout.addLayout(checkbox_layout)

    widget.darkfield_label = None
    widget.darkfield_selection = None

    return darkfield_layout


def add_flatfield_section(widget):
    """
    Create and return the flatfield-related UI components.
    """
    flatfield_layout = QVBoxLayout()

    checkbox_layout = QHBoxLayout()
    checkbox_layout.addWidget(QLabel("Flatfield:"))
    widget.flatfield_checkbox = QCheckBox()
    widget.flatfield_checkbox.stateChanged.connect(
        lambda state: toggle_field_widgets(widget, state, flatfield_layout, 'flatfield_label', 'flatfield_selection', 'Select flatfield:')
    )
    checkbox_layout.addWidget(widget.flatfield_checkbox)

    flatfield_layout.addLayout(checkbox_layout)

    widget.flatfield_label = None
    widget.flatfield_selection = None

    return flatfield_layout


def add_preprocessing_section(widget):
    """
    Add preprocessing section to the widget.
    """
    group_box = QGroupBox("Preprocessing")
    layout = QVBoxLayout() 

    flatfield_layout = add_flatfield_section(widget)
    layout.addLayout(flatfield_layout)

    darkfield_layout = add_darkfield_section(widget)
    layout.addLayout(darkfield_layout)

    apply_button = QPushButton("Apply")
    apply_button.clicked.connect(lambda: call_preprocess(widget.experiment, widget.viewer, widget))
    layout.addWidget(apply_button)

    group_box.setLayout(layout)

    widget.layout().addWidget(group_box)

def add_energy_layout(widget):
    layout = QHBoxLayout()
    layout.addWidget(QLabel("Energy (keV):"))
    widget.energy_input = QLineEdit()
    widget.energy_input.setText(str(widget.experiment.energy) if widget.experiment.energy is not None else "")
    layout.addWidget(widget.energy_input)
    widget.variables_layout.addLayout(layout)

def add_pixel_size_layout(widget):
    layout = QHBoxLayout()
    layout.addWidget(QLabel("Pixel size (m):"))
    widget.pixel_size_input = QLineEdit()
    widget.pixel_size_input.setText(str(widget.experiment.pixel) if widget.experiment.pixel is not None else "")
    layout.addWidget(widget.pixel_size_input)
    widget.variables_layout.addLayout(layout)

def add_delta_beta_layout(widget):
    layout = QHBoxLayout()
    layout.addWidget(QLabel("Delta Beta Ratio:"))
    widget.db_input = QLineEdit()
    widget.db_input.setText(str(widget.experiment.db) if widget.experiment.db is not None else "")
    layout.addWidget(widget.db_input)
    widget.variables_layout.addLayout(layout)

def add_distance_object_detector_layout(widget):
    layout = QHBoxLayout()
    layout.addWidget(QLabel("Distance object-detector (m):"))
    widget.distance_object_detector_input = QLineEdit()
    widget.distance_object_detector_input.setText(str(widget.experiment.dist_object_detector) if widget.experiment.dist_object_detector is not None else "")
    layout.addWidget(widget.distance_object_detector_input)
    widget.variables_layout.addLayout(layout)

def add_unsharp_mask_layout(widget):
    layout = QHBoxLayout()
    layout.addWidget(QLabel("Unsharp mask → Sigma:"))
    widget.sigma_input = QLineEdit()
    widget.sigma_input.setText(str(widget.experiment.sigma) if widget.experiment.sigma is not None else "")
    layout.addWidget(widget.sigma_input)
    layout.addWidget(QLabel("Coefficient:"))
    widget.coeff_input = QLineEdit()
    widget.coeff_input.setText(str(widget.experiment.coeff) if widget.experiment.coeff is not None else "")
    layout.addWidget(widget.coeff_input)
    widget.variables_layout.addLayout(layout)

def toggle_paganin(widget, checked, layout):
    """
    Toggle the visibility of the energy input field and other parameters based on the Paganin checkbox state.
    """
    if checked == Qt.Checked:
        if not hasattr(widget, 'variables_layout'):
            widget.variables_layout = QVBoxLayout()
            layout.addLayout(widget.variables_layout)

        add_energy_layout(widget)
        add_pixel_size_layout(widget)
        add_distance_object_detector_layout(widget)
        add_delta_beta_layout(widget)
        add_unsharp_mask_layout(widget)

        widget.paganin_apply_button = QPushButton("Apply")
        widget.paganin_apply_button.clicked.connect(lambda: call_paganin(widget.experiment, widget.viewer, widget))
        layout.addWidget(widget.paganin_apply_button)

    else:
        if hasattr(widget, 'variables_layout') and widget.variables_layout:
            while widget.variables_layout.count():
                child = widget.variables_layout.takeAt(0)
                if child.widget():
                    child.widget().deleteLater()
            del widget.variables_layout

        if hasattr(widget, 'paganin_apply_button') and widget.paganin_apply_button:
            layout.removeWidget(widget.paganin_apply_button)
            widget.paganin_apply_button.deleteLater()
            del widget.paganin_apply_button

def add_paganin_section(widget):
    """
    Add Paganin section to the widget.
    """
    group_box = QGroupBox("Paganin")
    layout = QVBoxLayout()

    widget.paganin_checkbox = QCheckBox("Enable Paganin")
    widget.paganin_checkbox.stateChanged.connect(
        lambda state: toggle_paganin(widget, state, layout)
    )
    layout.addWidget(widget.paganin_checkbox)

    group_box.setLayout(layout)
    widget.layout().addWidget(group_box)

def add_double_flatfield_section(widget):
    """
    Add double flatfield section to the widget.
    """
    group_box = QGroupBox("Double Flatfield")
    layout = QVBoxLayout()

    widget.double_flatfield_checkbox = QCheckBox("Enable Double Flatfield")
    layout.addWidget(widget.double_flatfield_checkbox)

    group_box.setLayout(layout)
    widget.layout().addWidget(group_box)

def add_angles_section(widget):
    """
    Add angles section to the widget.
    """
    group_box = QGroupBox("Angles")
    layout = QVBoxLayout()

    widget.angles_checkbox = QCheckBox("Enable Angles (if half acquisition and .nxs file)")
    layout.addWidget(widget.angles_checkbox)

    group_box.setLayout(layout)
    widget.layout().addWidget(group_box)

def setup_standard_acquisition(widget, layout):
    """
    Configure the widgets for Standard Acquisition mode.
    """
    cleanup_half_acquisition(widget, layout)

    if not hasattr(widget, 'cor_min_label') or widget.cor_min_label is None:
        cor_layout = QHBoxLayout()

        widget.cor_min_label = QLabel("COR Min:")
        widget.cor_min_input = QLineEdit()
        widget.cor_min_input.setPlaceholderText("Enter minimum COR")
        widget.cor_min_input.setText(str(widget.experiment.cor_min) if widget.experiment.cor_min is not None else "")
        cor_layout.addWidget(widget.cor_min_label)
        cor_layout.addWidget(widget.cor_min_input)

        widget.cor_max_label = QLabel("COR Max:")
        widget.cor_max_input = QLineEdit()
        widget.cor_max_input.setPlaceholderText("Enter maximum COR")
        widget.cor_max_input.setText(str(widget.experiment.cor_max) if widget.experiment.cor_max is not None else "")
        cor_layout.addWidget(widget.cor_max_label)
        cor_layout.addWidget(widget.cor_max_input)

        layout.addLayout(cor_layout)

    if not hasattr(widget, 'cor_step_label') or widget.cor_step_label is None:
        cor_step_layout = QHBoxLayout()
        widget.cor_step_label = QLabel("COR Step:")
        widget.cor_step_input = QLineEdit()
        widget.cor_step_input.setPlaceholderText("Enter step size")
        widget.cor_step_input.setText(str(widget.experiment.cor_step) if widget.experiment.cor_step is not None else "")
        cor_step_layout.addWidget(widget.cor_step_label)
        cor_step_layout.addWidget(widget.cor_step_input)
        layout.addLayout(cor_step_layout)

    if not hasattr(widget, 'try_cor_button') or widget.try_cor_button is None:
        widget.try_cor_button = QPushButton("Try Center of Rotation")
        widget.try_cor_button.clicked.connect(lambda: call_standard_cor_test(widget.experiment, widget.viewer, widget))
        layout.addWidget(widget.try_cor_button)


def setup_half_acquisition(widget, layout):
    """
    Configure the widgets for Half Acquisition mode.
    """
    cleanup_standard_acquisition(widget, layout)

    if not hasattr(widget, 'global_button') or widget.global_button is None:
        button_layout = QHBoxLayout()

        widget.global_button = QPushButton("Find Global Center of Rotation")
        widget.global_button.clicked.connect(lambda: call_find_global_cor(widget.experiment, widget.viewer, widget))
        button_layout.addWidget(widget.global_button)

        layout.addLayout(button_layout)

    if not hasattr(widget, 'fenetre_label') or widget.fenetre_label is None:
        fenetre_layout = QHBoxLayout()
        widget.fenetre_label = QLabel("Delta Center of Rotation:")
        widget.cor_fenetre_input = QSpinBox()
        widget.cor_fenetre_input.setMinimum(1)
        widget.cor_fenetre_input.setMaximum(1000)
        widget.cor_fenetre_input.setValue(10)  
        widget.cor_fenetre_input.setValue(int(widget.experiment.cor_fenetre) if widget.experiment.cor_fenetre is not None else 10)
        fenetre_layout.addWidget(widget.fenetre_label)
        fenetre_layout.addWidget(widget.cor_fenetre_input)
        layout.addLayout(fenetre_layout)

    if not hasattr(widget, 'try_cor_button') or widget.try_cor_button is None:
        widget.try_cor_button = QPushButton("Try Center of Rotation")
        widget.try_cor_button.clicked.connect(lambda: call_half_cor_test(widget.experiment, widget.viewer, widget))
        layout.addWidget(widget.try_cor_button)


def cleanup_standard_acquisition(widget, layout):
    """
    Remove widgets related to Standard Acquisition.
    """
    for attr in ['try_cor_button', 'cor_min_label', 'cor_min_input', 'cor_max_label', 'cor_max_input', 'cor_step_label', 'cor_step_input']:
        if hasattr(widget, attr) and getattr(widget, attr) is not None:
            widget_attr = getattr(widget, attr)
            layout.removeWidget(widget_attr)
            widget_attr.deleteLater()
            setattr(widget, attr, None)
    if hasattr(widget, 'cor_layout') and widget.cor_layout is not None:
        while widget.cor_layout.count():
            child = widget.cor_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        del widget.cor_layout


def cleanup_half_acquisition(widget, layout):
    """
    Remove widgets related to Half Acquisition.
    """
    for attr in ['global_button', 'fenetre_label', 'cor_fenetre_input', 'try_cor_button']:
        if hasattr(widget, attr) and getattr(widget, attr) is not None:
            widget_attr = getattr(widget, attr)
            layout.removeWidget(widget_attr)
            widget_attr.deleteLater()
            setattr(widget, attr, None)
    if hasattr(widget, 'button_layout') and widget.button_layout is not None:
        while widget.button_layout.count():
            child = widget.button_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        del widget.button_layout


def toggle_center_of_rotation(widget, state, layout):
    """
    Toggle the visibility of the center of rotation input field based on the acquisition type.
    """
    if state == 0:  # Standard Acquisition
        setup_standard_acquisition(widget, layout)
    else:  # Half Acquisition
        setup_half_acquisition(widget, layout)

def add_center_of_rotation_section(widget):
    """
    Add center of rotation section to widget.
    """
    group_box = QGroupBox("Center of Rotation")
    layout = QVBoxLayout()

    acquisition_type_layout = QHBoxLayout()
    acquisition_type_label = QLabel("Acquisition Type:")
    widget.acquisition_type_selection = QComboBox()
    widget.acquisition_type_selection.addItems(["Standard Acquisition", "Half Acquisition"])
    acquisition_type_layout.addWidget(acquisition_type_label)
    acquisition_type_layout.addWidget(widget.acquisition_type_selection)

    widget.acquisition_type_selection.currentIndexChanged.connect(
        lambda state: toggle_center_of_rotation(widget, state, layout)
    )

    layout.addLayout(acquisition_type_layout)

    center_of_rotation_layout = QHBoxLayout()
    center_of_rotation_label = QLabel("Center of Rotation:")
    widget.center_of_rotation_input = QLineEdit()
    widget.center_of_rotation_input.setText(
        str(widget.experiment.center_of_rotation) if widget.experiment.center_of_rotation is not None else ""
    )
    center_of_rotation_layout.addWidget(center_of_rotation_label)
    center_of_rotation_layout.addWidget(widget.center_of_rotation_input)
    layout.addLayout(center_of_rotation_layout)

    group_box.setLayout(layout)
    widget.layout().addWidget(group_box)

    toggle_center_of_rotation(widget, widget.acquisition_type_selection.currentIndex(), layout)


# ─── Reconstruction algorithm section ────────────────────────────────────────

# Base algorithm names — _CUDA suffix is appended automatically when GPU is selected
# FP (Forward Projection) is excluded: it projects a volume → sinogram, not the reverse
_ALGO_LIST   = ['FBP', 'BP', 'SIRT', 'SART', 'CGLS']
_FBP_FILTERS = ['ram-lak', 'shepp-logan', 'cosine', 'hamming', 'hann',
                 'tukey', 'lanczos', 'triangular', 'gaussian', 'parzen']


def _row_widget(*items):
    """Wrap a list of QWidgets in a container QWidget with a QHBoxLayout."""
    from qtpy.QtWidgets import QWidget as _QWidget
    container = _QWidget()
    row = QHBoxLayout(container)
    row.setContentsMargins(0, 0, 0, 0)
    for item in items:
        row.addWidget(item)
    return container


_RECON_WIDGET_ATTRS = (
    'recon_filter_combo', 'recon_iterations_input',
    'recon_min_input', 'recon_max_input',
)


def _clear_recon_params(widget):
    layout = widget._recon_params_layout
    while layout.count():
        item = layout.takeAt(0)
        if item.widget():
            item.widget().deleteLater()
    # Nullify Python references so update_parameters won't access deleted C++ objects
    for attr in _RECON_WIDGET_ATTRS:
        setattr(widget, attr, None)


def _build_recon_params(widget):
    """Populate _recon_params_layout for the currently selected algorithm."""
    algo   = widget.recon_algo_combo.currentText()   # base name, e.g. 'FBP'
    layout = widget._recon_params_layout
    exp    = widget.experiment

    if algo == 'FBP':
        widget.recon_filter_combo = QComboBox()
        widget.recon_filter_combo.addItems(_FBP_FILTERS)
        saved = exp.recon_filter_type or 'ram-lak'
        idx = widget.recon_filter_combo.findText(saved)
        widget.recon_filter_combo.setCurrentIndex(max(0, idx))
        layout.addWidget(_row_widget(QLabel("Filter:"), widget.recon_filter_combo))

    elif algo in ('SIRT', 'SART', 'CGLS'):
        widget.recon_iterations_input = QSpinBox()
        widget.recon_iterations_input.setMinimum(1)
        widget.recon_iterations_input.setMaximum(10000)
        widget.recon_iterations_input.setValue(
            int(exp.recon_iterations) if exp.recon_iterations else 100)
        layout.addWidget(
            _row_widget(QLabel("Iterations:"), widget.recon_iterations_input))

        if algo == 'SIRT':
            widget.recon_min_input = QLineEdit()
            widget.recon_min_input.setPlaceholderText("none")
            widget.recon_min_input.setText(
                str(exp.recon_min_constraint) if exp.recon_min_constraint is not None else "")
            widget.recon_max_input = QLineEdit()
            widget.recon_max_input.setPlaceholderText("none")
            widget.recon_max_input.setText(
                str(exp.recon_max_constraint) if exp.recon_max_constraint is not None else "")
            layout.addWidget(_row_widget(
                QLabel("Min constraint:"), widget.recon_min_input,
                QLabel("Max:"),           widget.recon_max_input))


def _on_recon_algo_changed(widget):
    _clear_recon_params(widget)
    _build_recon_params(widget)


def add_reconstruction_section(widget):
    """Add the Reconstruction Algorithm section to the widget."""
    group_box = QGroupBox("Reconstruction Algorithm")
    layout    = QVBoxLayout()

    # Algorithm selector (base names, no _CUDA suffix)
    widget.recon_algo_combo = QComboBox()
    widget.recon_algo_combo.addItems(_ALGO_LIST)
    saved_algo = widget.experiment.recon_algo or 'FBP'
    idx = widget.recon_algo_combo.findText(saved_algo)
    widget.recon_algo_combo.setCurrentIndex(max(0, idx))
    layout.addWidget(_row_widget(QLabel("Algorithm:"), widget.recon_algo_combo))

    # GPU / CPU checkbox
    widget.recon_gpu_checkbox = QCheckBox("Use GPU (CUDA)")
    saved_gpu = widget.experiment.recon_gpu
    # QSettings deserializes booleans as strings 'true'/'false'
    if isinstance(saved_gpu, str):
        saved_gpu = saved_gpu.lower() == 'true'
    widget.recon_gpu_checkbox.setChecked(saved_gpu if saved_gpu is not None else True)
    layout.addWidget(widget.recon_gpu_checkbox)

    # Dynamic parameter area
    widget._recon_params_layout = QVBoxLayout()
    layout.addLayout(widget._recon_params_layout)

    widget.recon_algo_combo.currentTextChanged.connect(
        lambda _: _on_recon_algo_changed(widget))

    group_box.setLayout(layout)
    widget.layout().addWidget(group_box)

    _build_recon_params(widget)


# ─── Process section ──────────────────────────────────────────────────────────

def add_process_section(widget):
    """
    Add process section to the widget.
    """
    group_box = QGroupBox("Process")
    layout = QVBoxLayout()

    process_one_slice_button = QPushButton("Process one slice")
    process_one_slice_button.clicked.connect(lambda: call_process_one_slice(widget.experiment, widget.viewer, widget))
    layout.addWidget(process_one_slice_button)

    process_all_slices_button = QPushButton("Process all slices")
    process_all_slices_button.clicked.connect(lambda: call_process_all_slices(widget.experiment, widget.viewer, widget))
    layout.addWidget(process_all_slices_button)

    group_box.setLayout(layout)
    widget.layout().addWidget(group_box)