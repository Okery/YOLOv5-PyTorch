import unittest
from pathlib import Path
from unittest import TestCase

import torch
from PIL import Image
from torchvision import transforms

import yolo
import model_utils


class TestMiniaturise(TestCase):
    IMAGE_SIZE = (640, 640)
    SCORE_THRESHOLD = 0.3
    NUM_CLASSES = 80
    BATCH_SIZE = 8
    DEVICE = "cuda"
    path = "./yolov5s_official_2cf45318.pth"
    dataPath = "./mini.pth"

    def test_inference(self):
        model = self._get_model()
        results, losses = model([self._get_image()])
        assert results[0].boxes[0][0].item() == 489.06707763671875
        assert results[0].boxes[0][1].item() == 15.00164794921875
        assert results[0].boxes[0][2].item() == 919.7750854492188
        assert results[0].boxes[0][3].item() == 815.4398193359375
        assert results[0].labels[0].item() == 15
        assert results[0].scores[0].item() == 0.8907631635665894

    def test_miniaturisation(self):
        model = self._get_model()
        model_utils.pytorch_miniaturise_and_save(model, self.IMAGE_SIZE, self.BATCH_SIZE, self.dataPath, True)

    def _get_image(self, path="images/cat.jpeg"):
        return transforms.ToTensor()(Image.open(path).convert("RGB")).to(self.DEVICE)

    def _get_model(self):
        model = yolo.YOLOv5(
            self.NUM_CLASSES,
            img_sizes=self.IMAGE_SIZE,
            score_thresh=self.SCORE_THRESHOLD,
        ).to(self.DEVICE)
        model.eval()
        model.load_state_dict(torch.load(Path(self.path)))

        return model


if __name__ == "__main__":
    unittest.main()
