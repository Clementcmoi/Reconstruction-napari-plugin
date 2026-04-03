from qtpy.QtWidgets import QDialog, QVBoxLayout, QLabel, QApplication, QMainWindow, QSlider, QWidget
from qtpy.QtCore import Qt, QThread, Signal

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class WorkerThread(QThread):
    finished = Signal(object)
    error = Signal(str)

    def __init__(self, func):
        super().__init__()
        self.func = func

    def run(self):
        try:
            result = self.func()
            self.finished.emit(result)
        except Exception as e:
            import traceback
            self.error.emit(traceback.format_exc())


def create_processing_dialog(parent, message="Processing..."):
    """
    Create and display a dialog with a message to indicate that processing is ongoing.
    """
    dialog = QDialog(parent)
    dialog.setWindowTitle("Processing")
    layout = QVBoxLayout()
    label = QLabel(message)
    layout.addWidget(label)
    dialog.setLayout(layout)
    dialog.setFixedSize(200, 100)
    dialog.show()
    QApplication.processEvents()
    return dialog

class PlotWindow(QMainWindow):
    def __init__(self, plot_data, cor_values):
        super().__init__()
        self.plot_data = plot_data
        self.cor_values = cor_values
        self.current_index = 0

        self.setWindowTitle("Plot Viewer")
        self.setGeometry(100, 100, 800, 600)

        self.canvas = FigureCanvas(Figure(figsize=(5, 3)))
        self.ax = self.canvas.figure.subplots()
        self.update_plot()

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(len(plot_data) - 1)
        self.slider.setValue(self.current_index)
        self.slider.valueChanged.connect(self.slider_changed)

        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        layout.addWidget(self.slider)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def update_plot(self):
        self.ax.clear()
        self.ax.plot(self.plot_data[self.current_index])
        self.ax.set_title(f"Plot {self.current_index + 1} (COR: {self.cor_values[self.current_index]:.2f})")
        self.canvas.draw()

    def slider_changed(self, value):
        self.current_index = value
        self.update_plot()