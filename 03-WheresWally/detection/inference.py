import cv2
import torch

from .model import get_model
from .transform import get_transform


class Inference():
    def __init__(self, weights_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = get_model(pretrained=False)
        self.model = self.model.to(self.device)
        self.model.eval()

        data = torch.load(weights_path, map_location=self.device)
        self.model.load_state_dict(data)

        self.transform = get_transform()

    def run_on_image(self, image):
        return self.run_on_image_list([image])[0]

    def run_on_image_list(self, image_list):
        images = []

        for i in image_list:
            i = cv2.cvtColor(i, cv2.COLOR_BGR2RGB)
            i = torch.from_numpy(i.astype("float32"))
            i /= 255
            i = i.permute(2, 0, 1)
            i = i.to(self.device)
            images.append(i)

        with torch.no_grad():
            output = self.model(images)

        result = []
        for o in output:
            r = {"boxes": o["boxes"].cpu().numpy(),
                 "scores": o["scores"].cpu().numpy()}
            result.append(r)

        return result
