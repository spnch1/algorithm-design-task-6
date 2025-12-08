import sys
import numpy as np
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QSpinBox, QPushButton, QTabWidget, QTextEdit)
from PyQt6.QtCore import QThread, pyqtSignal
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from solver import BeeTSP, ParameterTuner, NUM_CITIES, calculate_cost

class SolverThread(QThread):
    progress = pyqtSignal(int, float)
    finished = pyqtSignal(object, list)

    def __init__(self, bees, sites, iterations):
        super().__init__()
        self.solver = BeeTSP(bees, sites, iterations)

    def run(self):
        best_path, history = self.solver.run(self.progress_callback)
        self.finished.emit(best_path, history)

    def progress_callback(self, it, cost):
        self.progress.emit(it, cost)

class TunerThread(QThread):
    log_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(dict)

    def __init__(self):
        super().__init__()
        self.tuner = ParameterTuner()
    
    def run(self):
        best_params = self.tuner.tune(self.log_callback)
        self.finished_signal.emit(best_params)

    def log_callback(self, message):
        self.log_signal.emit(message)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Bee Algorithm TSP Visualizer & Tuner")
        self.resize(1100, 750)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QHBoxLayout(central_widget)

        controls = QWidget()
        controls.setFixedWidth(300)
        c_layout = QVBoxLayout(controls)
        
        c_layout.addWidget(QLabel("<b>Configuration</b>"))
        
        c_layout.addWidget(QLabel("Number of Bees:"))
        self.spin_bees = QSpinBox()
        self.spin_bees.setRange(10, 500)
        self.spin_bees.setValue(100)
        c_layout.addWidget(self.spin_bees)

        c_layout.addWidget(QLabel("Number of Sites:"))
        self.spin_sites = QSpinBox()
        self.spin_sites.setRange(2, 200)
        self.spin_sites.setValue(25)
        c_layout.addWidget(self.spin_sites)

        c_layout.addWidget(QLabel("Iterations:"))
        self.spin_iter = QSpinBox()
        self.spin_iter.setRange(100, 5000)
        self.spin_iter.setValue(1000)
        c_layout.addWidget(self.spin_iter)

        self.btn_run = QPushButton("Run Algorithm")
        self.btn_run.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 10px;")
        self.btn_run.clicked.connect(self.start_solver)
        c_layout.addWidget(self.btn_run)

        c_layout.addSpacing(20)
        c_layout.addWidget(QLabel("<b>Auto-Tuning</b>"))
        self.btn_tune = QPushButton("Start Parameter Tuning")
        self.btn_tune.setStyleSheet("background-color: #2196F3; color: white; font-weight: bold; padding: 10px;")
        self.btn_tune.clicked.connect(self.start_tuner)
        c_layout.addWidget(self.btn_tune)

        c_layout.addStretch()
        
        self.lbl_status = QLabel("Ready")
        c_layout.addWidget(self.lbl_status)
        self.lbl_result = QLabel("Best Dist: N/A")
        self.lbl_result.setStyleSheet("font-size: 14px; font-weight: bold;")
        c_layout.addWidget(self.lbl_result)

        layout.addWidget(controls)

        self.tabs = QTabWidget()
        
        self.fig_map = Figure(figsize=(5, 4), dpi=100)
        self.canvas_map = FigureCanvas(self.fig_map)
        self.ax_map = self.fig_map.add_subplot(111)
        self.ax_map.set_title("Best Path Found")
        self.tabs.addTab(self.canvas_map, "Map View")

        self.fig_conv = Figure(figsize=(5, 4), dpi=100)
        self.canvas_conv = FigureCanvas(self.fig_conv)
        self.ax_conv = self.fig_conv.add_subplot(111)
        self.ax_conv.set_title("Convergence (Cost vs Iterations)")
        self.ax_conv.set_xlabel("Iterations")
        self.ax_conv.set_ylabel("Distance")
        self.tabs.addTab(self.canvas_conv, "Convergence Graph")

        self.txt_logs = QTextEdit()
        self.txt_logs.setReadOnly(True)
        self.tabs.addTab(self.txt_logs, "Tuning Logs")

        layout.addWidget(self.tabs)

        self.city_coords = np.random.rand(NUM_CITIES, 2) * 100
        self.plot_map(np.arange(NUM_CITIES))

    def plot_map(self, path):
        self.ax_map.clear()
        self.ax_map.scatter(self.city_coords[:,0], self.city_coords[:,1], c='red', s=10, zorder=2)
        
        if path is not None:
            ordered_coords = self.city_coords[path]
            ordered_coords = np.vstack([ordered_coords, ordered_coords[0]])
            self.ax_map.plot(ordered_coords[:,0], ordered_coords[:,1], c='blue', alpha=0.6, zorder=1)
            self.ax_map.set_title(f"Map (Dist: {calculate_cost(path)})")
        
        self.canvas_map.draw()

    def start_solver(self):
        self.set_controls_enabled(False)
        self.lbl_status.setText("Optimizing...")
        
        bees = self.spin_bees.value()
        sites = self.spin_sites.value()
        iterations = self.spin_iter.value()

        self.worker = SolverThread(bees, sites, iterations)
        self.worker.progress.connect(self.update_progress)
        self.worker.finished.connect(self.on_finished)
        self.worker.start()

    def update_progress(self, it, cost):
        self.lbl_status.setText(f"Iter: {it} | Cost: {cost}")

    def on_finished(self, best_path, history):
        self.set_controls_enabled(True)
        self.lbl_status.setText("Done!")
        self.lbl_result.setText(f"Best Dist: {history[-1]}")
        
        self.plot_map(best_path)
        
        self.ax_conv.clear()
        self.ax_conv.plot(history)
        self.ax_conv.set_title("Convergence")
        self.ax_conv.grid(True)
        self.canvas_conv.draw()

    def start_tuner(self):
        self.set_controls_enabled(False)
        self.lbl_status.setText("Tuning Parameters...")
        self.tabs.setCurrentWidget(self.txt_logs)
        self.txt_logs.clear()
        
        self.tuner_thread = TunerThread()
        self.tuner_thread.log_signal.connect(self.append_log)
        self.tuner_thread.finished_signal.connect(self.on_tuning_finished)
        self.tuner_thread.start()

    def append_log(self, text):
        self.txt_logs.moveCursor(self.txt_logs.textCursor().MoveOperation.End)
        self.txt_logs.insertPlainText(text)

    def on_tuning_finished(self, best_params):
        self.set_controls_enabled(True)
        self.lbl_status.setText("Tuning Complete")
        self.append_log(f"\nOptimization Finished! Best Parameters: {best_params}\n")
        
        self.spin_bees.setValue(best_params['num_bees'])
        self.spin_sites.setValue(best_params['num_sites'])
        self.spin_iter.setValue(best_params['limit'])

    def set_controls_enabled(self, enabled):
        self.btn_run.setEnabled(enabled)
        self.btn_tune.setEnabled(enabled)
        self.spin_bees.setEnabled(enabled)
        self.spin_sites.setEnabled(enabled)
        self.spin_iter.setEnabled(enabled)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())