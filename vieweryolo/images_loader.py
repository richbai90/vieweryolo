
import os
from pathlib import Path

import cv2
import yaml


class ImagesLoader:
    def __init__(self, dataset_path, mode="train"):
        datafile_name = "data.yaml"
        datafile_path = Path(dataset_path) / datafile_name
        self._dataset_path = Path(dataset_path)
        data = self._load_data(datafile_path)
        self._mode = mode
        self._parse_data(data)

    def _load_data(self, dataset_path):
        with open(dataset_path, "r") as f:
            data = yaml.load(f, Loader=yaml.FullLoader)
        return data

    def _parse_data(self, data):
        self._names = data["names"]

        self._images_train_path = data["train"]
        self._images_val_path = data["val"]

        self._annotations_train_path = self._images_train_path.replace(
            "images", "labels"
        )
        self._annotations_val_path = self._images_val_path.replace("images", "labels")

        self._images_train_path = self._dataset_path / self._images_train_path
        self._images_val_path = self._dataset_path / self._images_val_path

        self._annotations_train_path = self._dataset_path / self._annotations_train_path
        self._annotations_val_path = self._dataset_path / self._annotations_val_path

    def _get_val_or_train(self):
        if self._mode == "train":
            return self._images_train_path, self._annotations_train_path
        elif self._mode == "val":
            return self._images_val_path, self._annotations_val_path

    def _read_image(self, image_path):
        image = cv2.imread(image_path)
        if image is None:
            return image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def _read_annotation(self, annotation_path):
        with open(annotation_path, "r") as f:
            return [[float(x) for x in line.strip().split()] for line in f]

    @property
    def names(self):
        return self._names

    def __getitem__(self, index):
        image_path, annotation_path = self._get_val_or_train()
        image_path = image_path / 'images'
        annotation_path = annotation_path / 'labels'
        images = os.listdir(image_path)
        image_path = Path(image_path) / images[index]
        annotation_path = Path(annotation_path) / images[index].replace(
            ".jpg", ".txt"
        ).replace(".png", ".txt")
        image_path = str(image_path)
        annotation_path = str(annotation_path)

        image = self._read_image(image_path)
        annotation = self._read_annotation(annotation_path)
        return image, annotation

    def __len__(self):
        return len(os.listdir(self._get_val_or_train()[0]))


if __name__ == "__main__":
    im = ImagesLoader("C:Users/andrey/Desktop/slrbot/ml/fixed_dataset", "train")

    print(im[0])
