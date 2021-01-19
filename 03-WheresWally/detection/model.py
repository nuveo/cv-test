import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def get_model(pretrained=True):
    """
    Return a faster rcnn model. Usually, pretrained is True during train and False during test.
    :param pretrained: If True, the weights (except box_predictor) are pretrained in imagenet.
    :return: A faster rcnn model
    """
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=pretrained, pretrained_backbone=pretrained, max_size=500)
    num_classes = 2
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model
