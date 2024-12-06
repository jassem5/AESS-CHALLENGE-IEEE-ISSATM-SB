import torch.nn as nn

class ResNet9(nn.Module):
    '''
    ResNet9 model architecture definition
    '''

    def __init__(self, in_channels: int, num_diseases: int) -> None:
        super().__init__()

        self.conv1 = self.convolution_block(in_channels, 64)
        self.conv2 = self.convolution_block(64, 128, pool=True) 
        self.res1 = nn.Sequential(self.convolution_block(128, 128), self.convolution_block(128, 128))

        self.conv3 = self.convolution_block(128, 256, pool=True)
        self.conv4 = self.convolution_block(256, 512, pool=True)
        self.res2 = nn.Sequential(self.convolution_block(512, 512), self.convolution_block(512, 512))

        self.classifier = nn.Sequential(nn.MaxPool2d(4), nn.Flatten(), nn.Linear(512, num_diseases))


    @staticmethod
    def convolution_block(in_channels: int, out_channels: int, pool: bool = False):
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        if pool:
            layers.append(nn.MaxPool2d(4))
        return nn.Sequential(*layers)


    def forward(self, input_tensor):
        '''
        Do inference on an input tensor (image)
        '''
        out = self.conv1(input_tensor)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.classifier(out)
        return out