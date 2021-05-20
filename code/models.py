import torch
import torch.nn as nn
import torchvision.models as models


# Resource: https://pytorch.org/docs/stable/torchvision/models.html
# Resource: https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html

# Modifying the fully connected layers.
class FullyConnec(nn.Module):

    def __init__(self):
        super(FullyConnec, self).__init__()

        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, 10)
        self.dropout = nn.Dropout(p=0.7)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(x)
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)

        return x

# Keeps autograd=True for the newly added layers and the rest becomes False. That is, weights won't be changed for the rest.
def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

# defines ResNet-50, ResNet-101, ResNet-152, and inception_v3.
def define_model(model_name, use_pretrained=True):
    if model_name == "resnet":

        model_ft = models.resnet50(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extracting=True)
        in_feature = model_ft.fc.in_features
        model_ft.fc = nn.Linear(in_feature, 1024)
        model_ft = nn.Sequential(model_ft, FullyConnec())
        # input_size = 224

        return model_ft

    elif model_name == "resnet1":

        model_ft = models.resnet152(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extracting=True)
        in_feature = model_ft.fc.in_features
        model_ft.fc = nn.Linear(in_feature, 1024)
        model_ft = nn.Sequential(model_ft, FullyConnec())
        # input_size = 224

        return model_ft

    elif model_name == "resnet2":

        model_ft = models.resnet101(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extracting=True)
        in_feature = model_ft.fc.in_features
        model_ft.fc = nn.Linear(in_feature, 1024)
        model_ft = nn.Sequential(model_ft, FullyConnec())
        # input_size = 224

        return model_ft

    elif model_name == "inception":

        model_ft = models.inception_v3(pretrained=use_pretrained)
        model_ft.conv = nn.Conv2d(5, 32, kernel_size=(3, 3), stride=(2, 2), bias=False)
        set_parameter_requires_grad(model_ft, feature_extracting=True)
        in_feature = model_ft.fc.in_features
        model_ft.fc = nn.Linear(in_feature, 1024)
        model_ft = nn.Sequential(model_ft, FullyConnec())
        # input_size = 299

        return model_ft

    elif model_name == "inception2":

        model_ft = models.inception_v3(pretrained=use_pretrained)
        model_ft.conv = nn.Conv2d(4, 32, kernel_size=(3, 3), stride=(2, 2), bias=False)
        set_parameter_requires_grad(model_ft, feature_extracting=True)
        in_feature = model_ft.fc.in_features
        model_ft.fc = nn.Linear(in_feature, 1024)
        model_ft = nn.Sequential(model_ft, FullyConnec())
        # input_size = 299

        return model_ft

    elif model_name == "faster_rcnn":
        model_ft = models.detection.fasterrcnn_resnet50_fpn(pretrained=True).to('cuda')
        in_features = model_ft.roi_heads.box_predictor.cls_score.in_features
        model_ft.roi_heads.box_predictor = models.detection.faster_rcnn.FastRCNNPredictor(in_features, 10)

        return model_ft

    else:
        Print("No model found.")    
        exit()


if __name__ == '__main__':
    model = define_model("resnet1")
    model.eval()
    model.cuda()
    inputs = torch.rand(8, 3, 224, 224).cuda()
    output = model(inputs)
    print(output.shape)
    print(model)
