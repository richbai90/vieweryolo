import sys
import os
import cv2
import numpy as np
from PyQt6.QtWidgets import (
    QApplication,
    QWidget,
    QLabel,
    QPushButton,
    QFileDialog,
    QVBoxLayout,
    QHBoxLayout,
    QSpacerItem,
    QSizePolicy,
    QCheckBox,
)
from PyQt6.QtGui import QPixmap, QImage, QIcon
from PyQt6.QtCore import Qt
from vieweryolo.images_loader import ImagesLoader
from shapely import Polygon


class YoloViewer(QWidget):
    def __init__(self):
        super().__init__()
        self.dataset_path = None
        self.im_loader = None
        self.index = 0
        self.initUI()

    def initUI(self):
        self.setWindowTitle("YOLO Viewer")
        self.setGeometry(100, 100, 1600, 1200)
        self.setStyleSheet("""
            QWidget {
                background-color: #3E3E3E;
                color: white;
                font-family: 'Roboto', sans-serif;
            }
            QPushButton {
                background-color: #444444;
                border: none;
                padding: 12px 24px;
                font-size: 16px;
                border-radius: 8px;
            }
            QPushButton:hover {
                background-color: #666666;
            }
            QLabel {
                background-color: #2E2E2E;
                border: 2px solid #555555;
                padding: 15px;
                border-radius: 12px;
                font-size: 18px;
                margin-bottom: 20px;
            }
            QCheckBox {
                font-size: 16px;
                margin-top: 10px;
                padding: 5px;
            }
        """)

        self.image_label = QLabel(self)
        self.image_label.setFixedSize(1600, 900)
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.select_button = QPushButton("Select data.yaml", self)
        self.select_button.clicked.connect(self.load_dataset)
        self.select_button.setIcon(QIcon("path/to/icon/file.png"))

        self.prev_button = QPushButton("Previous", self)
        self.prev_button.setIcon(QIcon("path/to/prev-icon.png"))
        self.prev_button.clicked.connect(self.prev_image)

        self.next_button = QPushButton("Next", self)
        self.next_button.setIcon(QIcon("path/to/next-icon.png"))
        self.next_button.clicked.connect(self.next_image)

        self.train_checkbox = QCheckBox("Use validation set", self)
        self.train_checkbox.stateChanged.connect(self.toggle_dataset)

        hbox = QHBoxLayout()
        hbox.addWidget(self.prev_button)
        hbox.addWidget(self.next_button)
        hbox.addItem(
            QSpacerItem(
                40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum
            )
        )

        vbox = QVBoxLayout()
        vbox.addWidget(self.select_button, alignment=Qt.AlignmentFlag.AlignTop)
        vbox.addWidget(self.train_checkbox, alignment=Qt.AlignmentFlag.AlignTop)
        vbox.addWidget(self.image_label, alignment=Qt.AlignmentFlag.AlignCenter)
        vbox.addLayout(hbox)
        vbox.addItem(
            QSpacerItem(
                20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding
            )
        )

        self.setLayout(vbox)

    def load_dataset(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select data.yaml", "", "YAML Files (*.yaml)"
        )
        if file_path:
            self.dataset_path = os.path.dirname(file_path)
            self.im_loader = ImagesLoader(
                self.dataset_path,
                "train" if not self.train_checkbox.isChecked() else "val",
            )
            self.index = 0
            self.show_image()

    def toggle_dataset(self):
        if self.dataset_path:
            subset = "val" if self.train_checkbox.isChecked() else "train"
            self.im_loader = ImagesLoader(self.dataset_path, subset)
            self.index = 0
            self.show_image()

    def yolo_to_abs(self, image, annotation):
        height, width, _ = image.shape
        # Create an empty segmask as default
        segmask = np.array([])
        class_id, x_center, y_center, box_width, box_height, *others = annotation
        if len(others):
            segmask = np.array(list(others)).reshape(-1, 2) * np.array([width, height])

        x_center *= width
        y_center *= height
        box_width *= width
        box_height *= height
        x1 = int(x_center - box_width / 2)
        y1 = int(y_center - box_height / 2)
        x2 = int(x_center + box_width / 2)
        y2 = int(y_center + box_height / 2)
        return int(class_id), x1, y1, x2, y2, segmask

    def draw_boxes(self, image, annotations, image_label_size):
        names = self.im_loader.names
        np.random.seed(42)
        colors = list(np.random.randint(0, 256, (len(names), 3)))

        for ann in annotations:
            class_id, x1, y1, x2, y2, mask = self.yolo_to_abs(image, ann)
            cv2.rectangle(
                image, (x1, y1), (x2, y2), tuple(map(int, colors[class_id])), 3
            )
            cv2.putText(
                image,
                names[class_id],
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                tuple(map(int, colors[class_id])),
                2,
            )

            if mask.size:
                poly = Polygon(mask)
                pts = np.array(poly.exterior.coords, dtype=np.int32).reshape(-1, 1, 2)
                cv2.fillPoly(image, pts=pts, color=(0, 255, 0))

        h, w, c = image.shape
        if h > image_label_size.height() or w > image_label_size.width():
            image = cv2.resize(
                image,
                (0, 0),
                fx=image_label_size.width() / w,
                fy=image_label_size.height() / h,
            )

        return image

    def show_image(self):
        if self.im_loader and 0 <= self.index < len(self.im_loader):
            image, annotations = self.im_loader[self.index]
            if image is None or annotations is None:
                image = np.zeros((640, 480, 3), dtype=np.uint32)
            image_label_size = self.image_label.size()

            image = self.draw_boxes(image, annotations, image_label_size)
            height, width, channel = image.shape
            bytes_per_line = 3 * width
            qimg = QImage(
                image.data, width, height, bytes_per_line, QImage.Format.Format_RGB888
            ).rgbSwapped()
            pixmap = QPixmap.fromImage(qimg)
            self.image_label.setPixmap(pixmap)

    def prev_image(self):
        if self.im_loader and self.index > 0:
            self.index -= 1
            self.show_image()

    def next_image(self):
        if self.im_loader and self.index < len(self.im_loader) - 1:
            self.index += 1
            self.show_image()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_A:
            self.prev_image()
        elif event.key() == Qt.Key.Key_D:
            self.next_image()


def main():
    app = QApplication(sys.argv)
    viewer = YoloViewer()
    viewer.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
