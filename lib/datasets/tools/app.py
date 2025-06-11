import pickle
def read_pkl(pkl_file):  # 读取pkl
    my_data = pickle.load(open(pkl_file, 'rb'))
    return my_data

import sys
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QFileDialog, QTextEdit,
    QVBoxLayout, QHBoxLayout, QGridLayout, QGroupBox, QComboBox, QProgressBar
)
from PyQt5.QtGui import QPixmap,QFont
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QListWidget, QSizePolicy
import os  # 用于列出目录内容


class DomainDetectionUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("面向不同数据可用性条件的遥感影像跨域目标检测原型系统")
        self.setGeometry(100, 100, 1200, 800)
        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout()

        # Title
        title = QLabel("面向不同数据可用性条件的遥感影像跨域目标检测原型系统")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 20px; font-weight: bold; margin: 10px;")
        main_layout.addWidget(title)

        # === Top Layout ===
        top_layout = QHBoxLayout()

        # Dataset Management
        dataset_group = QGroupBox("数据集管理")
        dataset_layout = QVBoxLayout()
        self.src_btn = QPushButton("选择源域数据集")
        self.src_btn.clicked.connect(self.select_source_dataset)
        self.tgt_btn = QPushButton("选择目标域数据集")
        self.tgt_btn.clicked.connect(self.select_target_dataset)
        self.src_file_list = QListWidget()
        self.src_file_list.setMinimumHeight(150)
        self.tgt_file_list = QListWidget()
        self.tgt_file_list.setMinimumHeight(150)
        dataset_layout.addWidget(self.src_btn)
        dataset_layout.addWidget(self.src_file_list)
        dataset_layout.addWidget(self.tgt_btn)
        dataset_layout.addWidget(self.tgt_file_list)
        dataset_group.setLayout(dataset_layout)
        top_layout.addWidget(dataset_group, 1)

        # Detection Image
        image_group = QGroupBox("检测图像")
        image_layout = QVBoxLayout()
        self.image_label = QLabel()
        self.image_label.setFixedSize(500, 400)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("border: 1px solid gray; background-color: #f8f8f8;")
        self.load_img_btn = QPushButton("显示图像")
        self.load_img_btn.clicked.connect(self.load_image)
        image_layout.addWidget(self.image_label)
        image_layout.addWidget(self.load_img_btn)
        image_group.setLayout(image_layout)
        top_layout.addWidget(image_group, 2)

        # PR Curve
        pr_group = QGroupBox("PR Curve")
        pr_layout = QVBoxLayout()
        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.figure)
        self.plot_pr_button = QPushButton("显示 PR 曲线")
        self.plot_pr_button.clicked.connect(self.plot_pr_curve)
        pr_layout.addWidget(self.plot_pr_button)
        pr_layout.addWidget(self.canvas)
        pr_group.setLayout(pr_layout)
        top_layout.addWidget(pr_group, 2)

        main_layout.addLayout(top_layout)

        # === Middle Layout ===
        mid_layout = QGridLayout()

        # Model Configuration
        model_group = QGroupBox("模型配置")
        config_layout = QGridLayout()
        self.model_selector = QComboBox()
        self.model_selector.addItems([
            "域特定分块通道注意力",
            "多层面域扰动",
            "可学习风格适应"
        ])
        self.epoch_input = QComboBox()
        self.epoch_input.addItems(["7", "20", "50", "100"])
        self.lr_input = QComboBox()
        self.lr_input.addItems(["0.001", "0.0001", "0.00001"])
        self.batch_input = QComboBox()
        self.batch_input.addItems(["1", "2", "4", "8"])
        config_layout.addWidget(QLabel("域适应算法:"), 0, 0)
        config_layout.addWidget(self.model_selector, 0, 1)
        config_layout.addWidget(QLabel("Epochs:"), 1, 0)
        config_layout.addWidget(self.epoch_input, 1, 1)
        config_layout.addWidget(QLabel("Learning Rate:"), 2, 0)
        config_layout.addWidget(self.lr_input, 2, 1)
        config_layout.addWidget(QLabel("Batch Size:"), 3, 0)
        config_layout.addWidget(self.batch_input, 3, 1)
        model_group.setLayout(config_layout)
        mid_layout.addWidget(model_group, 0, 0)

        # Model Actions
        action_group = QGroupBox("Model Actions")
        action_layout = QVBoxLayout()
        self.train_btn = QPushButton("Train")
        self.train_btn.clicked.connect(self.train_model)
        self.eval_btn = QPushButton("Evaluate")
        self.eval_btn.clicked.connect(self.evaluate_model)
        self.detect_btn = QPushButton("Detect")
        self.detect_btn.clicked.connect(self.detect_objects)
        for btn in [self.train_btn, self.eval_btn, self.detect_btn]:
            action_layout.addWidget(btn)
        action_group.setLayout(action_layout)
        mid_layout.addWidget(action_group, 0, 1)

        # Save / Load Model
        model_io_group = QGroupBox("模型管理")
        io_layout = QVBoxLayout()
        self.save_btn = QPushButton("Save Model")
        self.save_btn.clicked.connect(self.save_model)
        self.load_btn = QPushButton("Load Model")
        self.load_btn.clicked.connect(self.load_model)
        io_layout.addWidget(self.save_btn)
        io_layout.addWidget(self.load_btn)
        model_io_group.setLayout(io_layout)
        mid_layout.addWidget(model_io_group, 0, 2)

        main_layout.addLayout(mid_layout)

        # === Bottom: Logs and Progress ===
        bottom_layout = QVBoxLayout()
        logs_group = QGroupBox("日志")
        logs_layout = QVBoxLayout()
        self.log_output = QTextEdit()
        self.log_output.setFont(QFont("Courier", 10))
        self.log_output.setReadOnly(True)
        self.log_output.setFixedHeight(100)
        logs_layout.addWidget(self.log_output)
        logs_group.setLayout(logs_layout)
        bottom_layout.addWidget(logs_group)

        self.status_label = QLabel("Status: Idle")
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        bottom_layout.addWidget(self.status_label)
        bottom_layout.addWidget(self.progress_bar)

        main_layout.addLayout(bottom_layout)
        self.setLayout(main_layout)

    # === Functional Methods ===
    def log(self, msg):
        self.log_output.append(f"[INFO] {msg}")
        self.status_label.setText(f"Status: {msg}")

    def select_source_dataset(self):
        path = QFileDialog.getExistingDirectory(self, "Select Source Dataset")
        if path:
            self.log(f"Source dataset selected: {path}")
            self.src_file_list.clear()
            try:
                files = os.listdir(path)
                self.src_file_list.addItems(sorted(files))
            except Exception as e:
                self.log(f"Error reading files: {e}")

    def select_target_dataset(self):
        path = QFileDialog.getExistingDirectory(self, "Select Target Dataset")
        if path:
            self.log(f"Target dataset selected: {path}")
            self.tgt_file_list.clear()
            try:
                files = os.listdir(path)
                self.tgt_file_list.addItems(sorted(files))
            except Exception as e:
                self.log(f"Error reading files: {e}")

    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Images (*.png *.jpg *.bmp)")
        if file_path:
            pixmap = QPixmap(file_path).scaled(500, 400, Qt.KeepAspectRatio)
            self.image_label.setPixmap(pixmap)
            self.log(f"Image loaded: {file_path}")

    def train_model(self):
        self.log("Training started...")
        self.progress_bar.setValue(20)

    def evaluate_model(self):
        self.log("Evaluation started...")
        self.progress_bar.setValue(50)

    def detect_objects(self):
        self.log("Detection in progress...")
        self.progress_bar.setValue(80)

    def save_model(self):
        self.log("Model saved.")
        self.progress_bar.setValue(90)

    def load_model(self):
        self.log("Model loaded.")
        self.progress_bar.setValue(100)

    def plot_pr_curve(self):
        workdir = 'C:\\Users\\19331\\Documents\\dataset\\ISPRSPotsdam\\PR\\'
        prfile2 = workdir + 'DSCR_Car_pr.pkl'
        try:
            pr2 = read_pkl(prfile2)
            recall = pr2['rec']
            precision = pr2['prec']
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            ax.plot(recall, precision, marker='o', label="PDSCR")
            ax.set_title("Precision-Recall on ISPRSPotsdam")
            ax.set_xlabel("Recall")
            ax.set_ylabel("Precision")
            ax.legend()
            self.canvas.draw()
            self.log("PR curve plotted.")
        except Exception as e:
            self.log(f"Failed to plot PR curve: {e}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    ui = DomainDetectionUI()
    ui.show()
    sys.exit(app.exec_())


# def plot_pr_curve(self):
#
#     # DSCR Potsdam
#     workdir = 'C:\\Users\\19331\\Documents\\dataset\\ISPRSVaihingen\\PR\\'
#     prfile2 = workdir + 'DSCR_Car_pr.pkl'
#     pr2 = read_pkl(prfile2)
#     recall = pr2['rec']
#     precision = pr2['prec']
#
#     self.figure.clear()
#     ax = self.figure.add_subplot(111)
#     ax.plot(recall, precision, marker='o', label="PDSCR")
#     ax.set_title("Precision-Recall on ISPRSVaihingen")
#     ax.set_xlabel("Recall")
#     ax.set_ylabel("Precision")
#     ax.legend()
#     self.canvas.draw()
#     self.log("已绘制 PR 曲线")




