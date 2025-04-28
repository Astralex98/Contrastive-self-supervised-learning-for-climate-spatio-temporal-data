import torch
import torchvision

class ResNet(torch.nn.Module):
    
    def __init__(self):
        super().__init__()

        # Embedding size of ResNet
        self.C = 512

        # Encoder
        backbone = torchvision.models.resnet18(pretrained=False)

        # Modifying first layer to allow processing one-dimensional images
        conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
                                   bias=False)

        backbone.conv1 = conv1

        # Get feature extractor part
        self.feature_extractor = torch.nn.Sequential(*list(backbone.children())[:-1])
    
    def forward(self, x):

        # [B, T, R, H, W] -> [B*T*R, 1, H, W]
        B, T, R, H, W = x.shape
        x = x.view(-1, 1, H, W)

        # Feature extraction
        # [B*T*R, 1, H, W] -> [B*T*R, C, 1, 1] (C = 512 for ResNet)
        # Here B = 1
        features = self.feature_extractor(x)

        # [B*T*R, C, 1, 1] -> [R, T, C]
        features = features.view(R, T, self.C)
        return features