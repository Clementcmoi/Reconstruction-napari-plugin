from qtpy.QtCore import QSettings
from qtpy.QtWidgets import QLineEdit
import gc
import cupy as cp

class Experiment:
    def __init__(self): 
        self.settings = QSettings("Reco", "recoconfig")

        # Initialize parameters
        self.sample_images = None
        self.slice_idx = None
        self.bigdata = None

        self.darkfield = None
        self.flatfield = None

        self.energy = None
        self.pixel = None
        self.dist_object_detector = None
        self.db = None
        self.sigma = None
        self.coeff = None   

        self.double_flatfield = None

        self.angle_between_projections = None

        self.acquisition_type = None
        self.center_of_rotation = None
        self.cor_min = None
        self.cor_max = None
        self.cor_step = None
        self.cor_fenetre = None
        self.batch_size = 50

        # Reconstruction algorithm
        self.recon_algo = 'FBP'
        self.recon_gpu = True
        self.recon_filter_type = 'ram-lak'
        self.recon_iterations = 100
        self.recon_min_constraint = None
        self.recon_max_constraint = None

        self.load_settings()
    
    def load_settings(self):
        """
        Load settings from QSettings into the experiment attributes.
        """
        for attr in vars(self):
            if attr not in ["settings"]:
                value = self.settings.value(attr, getattr(self, attr))
                setattr(self, attr, value)

    def save_settings(self, parameters_to_save=None):
        """
        Save only the specified parameters to settings. If no parameters are specified, save all.
        """
        if not hasattr(self, 'settings'):
            raise AttributeError("Instance has no 'settings' attribute.")

        saved_parameters = {}
        for attr in vars(self):
            if attr not in ["settings"]:
                if parameters_to_save is None or attr in parameters_to_save:
                    self.settings.setValue(attr, getattr(self, attr))
                    saved_parameters[attr] = getattr(self, attr)

        print(f"Saved Parameters: {saved_parameters}")  # Debug print to confirm saved parameters
        return saved_parameters



    def update_parameters(self, widget, parameters_to_update=None):
        """
        Update the parameters based on the widget values, only for the specified parameters.
        """
        try:
            if parameters_to_update is None or "sample_images" in parameters_to_update:
                self.sample_images = widget.sample_selection.currentText()
            if parameters_to_update is None or "slice_idx" in parameters_to_update:
                self.slice_idx = int(widget.slice_selection.value())
            if parameters_to_update is None or "bigdata" in parameters_to_update:
                self.bigdata = widget.bigdata_checkbox.isChecked()

            if parameters_to_update is None or "darkfield" in parameters_to_update:
                self.darkfield = widget.darkfield_selection.currentText() if widget.darkfield_checkbox.isChecked() else None
            if parameters_to_update is None or "flatfield" in parameters_to_update:
                self.flatfield = widget.flatfield_selection.currentText() if widget.flatfield_checkbox.isChecked() else None

            if parameters_to_update is None or "energy" in parameters_to_update:
                self.energy = float(widget.energy_input.text())
            if parameters_to_update is None or "pixel" in parameters_to_update:
                self.pixel = float(widget.pixel_size_input.text())
            if parameters_to_update is None or "dist_object_detector" in parameters_to_update:
                self.dist_object_detector = float(widget.distance_object_detector_input.text())
            if parameters_to_update is None or "db" in parameters_to_update:
                self.db = float(widget.db_input.text())
            if parameters_to_update is None or "sigma" in parameters_to_update:
                self.sigma = float(widget.sigma_input.text())
            if parameters_to_update is None or "coeff" in parameters_to_update:
                self.coeff = float(widget.coeff_input.text())

            if parameters_to_update is None or "double_flatfield" in parameters_to_update:
                self.double_flatfield = widget.double_flatfield_checkbox.isChecked()

            if parameters_to_update is None or "angle_between_projections" in parameters_to_update:
                self.angle_between_projections = widget.angles_checkbox.isChecked()

            if parameters_to_update is None or "acquisition_type" in parameters_to_update:
                self.acquisition_type = widget.acquisition_type_selection.currentIndex()
            if parameters_to_update is None or "center_of_rotation" in parameters_to_update:
                self.center_of_rotation = float(widget.center_of_rotation_input.text())

            if parameters_to_update is None or "cor_min" in parameters_to_update:
                self.cor_min = int(widget.cor_min_input.text())
            if parameters_to_update is None or "cor_max" in parameters_to_update:
                self.cor_max = int(widget.cor_max_input.text())
            if parameters_to_update is None or "cor_step" in parameters_to_update:
                self.cor_step = int(widget.cor_step_input.text())
            if parameters_to_update is None or "cor_fenetre" in parameters_to_update:
                self.cor_fenetre = int(widget.cor_fenetre_input.value())

            if parameters_to_update is None or "batch_size" in parameters_to_update:
                if hasattr(widget, 'batch_size_input') and widget.batch_size_input is not None:
                    self.batch_size = int(widget.batch_size_input.value())

            if parameters_to_update is None or "recon_algo" in parameters_to_update:
                if getattr(widget, 'recon_algo_combo', None) is not None:
                    self.recon_algo = widget.recon_algo_combo.currentText()
            if parameters_to_update is None or "recon_gpu" in parameters_to_update:
                if getattr(widget, 'recon_gpu_checkbox', None) is not None:
                    self.recon_gpu = widget.recon_gpu_checkbox.isChecked()
            if parameters_to_update is None or "recon_filter_type" in parameters_to_update:
                if getattr(widget, 'recon_filter_combo', None) is not None:
                    self.recon_filter_type = widget.recon_filter_combo.currentText()
            if parameters_to_update is None or "recon_iterations" in parameters_to_update:
                if getattr(widget, 'recon_iterations_input', None) is not None:
                    self.recon_iterations = int(widget.recon_iterations_input.value())
            if parameters_to_update is None or "recon_min_constraint" in parameters_to_update:
                if getattr(widget, 'recon_min_input', None) is not None and widget.recon_min_input.text():
                    self.recon_min_constraint = float(widget.recon_min_input.text())
                else:
                    self.recon_min_constraint = None
            if parameters_to_update is None or "recon_max_constraint" in parameters_to_update:
                if getattr(widget, 'recon_max_input', None) is not None and widget.recon_max_input.text():
                    self.recon_max_constraint = float(widget.recon_max_input.text())
                else:
                    self.recon_max_constraint = None

            self.save_settings(parameters_to_save=parameters_to_update)

        except ValueError as e:
            print(f"Error updating parameters: {e}")


class mpExperiment(Experiment):
    def __init__(self):
        self.settings = QSettings("Reco_mp", "recoconfig_mp")

        # Initialize parameters
        self.sample_images = None
        self.slice_idx = None
        self.bigdata = None
        self.darkfield = None  
        self.flatfield = None

        self.energy = None
        self.pixel = None
        self.dist_object_detector = None
        self.sigma = None
        self.coeff = None
        self.acquisition_type = None
        self.center_of_rotation = None

        self.step = None
        self.db = []
        self.threshold = []
        self.batch_size = 50

        self.load_settings()

    def load_settings(self):
        """
        Load settings from QSettings into the experiment attributes.
        """
        for attr in vars(self):
            if attr not in ["settings"]:
                value = self.settings.value(attr, getattr(self, attr))
                setattr(self, attr, value)

    def save_settings(self, parameters_to_save=None):
        """
        Save only the specified parameters to settings. If no parameters are specified, save all.
        """
        if not hasattr(self, 'settings'):
            raise AttributeError("Instance has no 'settings' attribute.")

        saved_parameters = {}
        for attr in vars(self):
            if attr not in ["settings"]:
                if parameters_to_save is None or attr in parameters_to_save:
                    self.settings.setValue(attr, getattr(self, attr))
                    saved_parameters[attr] = getattr(self, attr)

        print(f"Saved Parameters: {saved_parameters}")
        return saved_parameters
    
    def update_parameters(self, widget, parameters_to_update=None, paganin_index=None):
        """
        Update the parameters based on the widget values, only for the specified parameters.
        If paganin_index is specified, only update db/threshold up to this index (inclusive).
        """
        try:
            if parameters_to_update is None or "sample_images" in parameters_to_update:
                self.sample_images = widget.sample_selection.currentText()
            if parameters_to_update is None or "slice_idx" in parameters_to_update:
                self.slice_idx = int(widget.slice_selection.value())
            if parameters_to_update is None or "bigdata" in parameters_to_update:
                self.bigdata = widget.bigdata_checkbox.isChecked()

            if parameters_to_update is None or "darkfield" in parameters_to_update:
                self.darkfield = widget.darkfield_selection.currentText() if widget.darkfield_checkbox.isChecked() else None
            if parameters_to_update is None or "flatfield" in parameters_to_update:
                self.flatfield = widget.flatfield_selection.currentText() if widget.flatfield_checkbox.isChecked() else None

            if parameters_to_update is None or "energy" in parameters_to_update:
                self.energy = float(widget.energy_input.text())
            if parameters_to_update is None or "pixel" in parameters_to_update:
                self.pixel = float(widget.pixel_size_input.text())
            if parameters_to_update is None or "dist_object_detector" in parameters_to_update:
                self.dist_object_detector = float(widget.distance_object_detector_input.text())
            if parameters_to_update is None or "sigma" in parameters_to_update:
                self.sigma = float(widget.sigma_input.text())
            if parameters_to_update is None or "coeff" in parameters_to_update:
                self.coeff = float(widget.coeff_input.text())

            if parameters_to_update is None or "acquisition_type" in parameters_to_update:
                self.acquisition_type = widget.acquisition_type_selection.currentIndex()
            if parameters_to_update is None or "center_of_rotation" in parameters_to_update:
                self.center_of_rotation = float(widget.center_of_rotation_input.text())

            # Update step (number of Paganin sections)
            self.step = len(widget.paganin_sections)
            # Ensure db and threshold lists have the correct length
            self.db = [None] * self.step
            self.threshold = [None] * self.step
            # Determine how many sections to update
            last_index = self.step if paganin_index is None else paganin_index + 1
            for i, section in enumerate(widget.paganin_sections):
                if i < last_index:
                    db_input = section.findChildren(QLineEdit)[0]
                    threshold_input = section.findChildren(QLineEdit)[1]
                    try:
                        self.db[i] = float(db_input.text())
                    except Exception:
                        self.db[i] = None
                    try:
                        self.threshold[i] = float(threshold_input.text())
                    except Exception:
                        self.threshold[i] = None
                # Leave following values as None

            if parameters_to_update is None or "batch_size" in parameters_to_update:
                if hasattr(widget, 'batch_size_input') and widget.batch_size_input is not None:
                    self.batch_size = int(widget.batch_size_input.value())

            # Save settings
            self.save_settings(parameters_to_save=parameters_to_update)

        except ValueError as e:
            print(f"Error updating parameters: {e}")

