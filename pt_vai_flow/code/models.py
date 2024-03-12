import torch
import torch.nn as nn
import torch.nn.functional as F

class FSRCNN(nn.Module):
    def __init__(self, d, s, m, input_size, upscaling_factor, color_channels):
        super(FSRCNN, self).__init__()
    
        self.input_size = input_size
        self.upscaling_factor = upscaling_factor
        self.color_channels = color_channels

        # Feature extraction
        self.feature_extraction = nn.Sequential(
            nn.Conv2d(color_channels, d, kernel_size=5, padding=2),
            nn.ReLU()
        )

        # Shrinking
        self.shrinking = nn.Sequential(
            nn.Conv2d(d, s, kernel_size=1, padding=0),
            nn.ReLU()
        )

        # Mapping
        mapping_layers = []
        for _ in range(m):
            mapping_layers.extend([
                nn.Conv2d(s, s, kernel_size=3, padding=1),
                nn.ReLU()
            ])
        self.mapping = nn.Sequential(*mapping_layers)

        # Expanding
        self.expanding = nn.Sequential(
            nn.Conv2d(s, d, kernel_size=1, padding=0),
            nn.ReLU()
        )

        # Deconvolution
        self.deconvolution = nn.ConvTranspose2d(
            in_channels=d,
            out_channels=color_channels,
            kernel_size=9,
            stride=2,
            padding=4,
            output_padding=1
        )

    def forward(self, x):
        x = self.feature_extraction(x)
        x = self.shrinking(x)
        x = self.mapping(x)
        x = self.expanding(x)
        x = self.deconvolution(x)

        return x
       

class FCN8(nn.Module):
    def __init__(self, nClasses):
        super(FCN8, self).__init__()
        # Define layers
        self.block1_conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.block1_conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.block2_conv1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.block2_conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.block3_conv1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.block3_conv2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.block3_conv3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.block4_conv1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.block4_conv2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.block4_conv3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.block5_conv1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.block5_conv2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.block5_conv3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv7_4a = nn.Conv2d(512, nClasses, kernel_size=7, padding=3)
        self.conv7_4b = nn.Conv2d(nClasses, nClasses, kernel_size=1)
        self.conv7_4 = nn.ConvTranspose2d(nClasses, nClasses, kernel_size=4, stride=4, bias=False)
        self.pool4_11 = nn.Conv2d(512, nClasses, kernel_size=1)
        self.pool411_b = nn.Conv2d(nClasses, nClasses, kernel_size=1)
        self.pool411_2 = nn.ConvTranspose2d(nClasses, nClasses, kernel_size=2, stride=2, bias=False)
        self.pool3_11 = nn.Conv2d(256, nClasses, kernel_size=1)
        self.add_layer = nn.Conv2d(nClasses * 3, nClasses, kernel_size=1)
        self.conv8 = nn.ConvTranspose2d(nClasses, nClasses, kernel_size=8, stride=8, bias=False)

    def forward(self, x):
        # Block 1
        x = F.relu(self.block1_conv1(x))
        x = F.relu(self.block1_conv2(x))
        f1 = F.max_pool2d(x, kernel_size=2, stride=2)
        
        # Block 2
        x = F.relu(self.block2_conv1(f1))
        x = F.relu(self.block2_conv2(x))
        f2 = F.max_pool2d(x, kernel_size=2, stride=2)
        
        # Block 3
        x = F.relu(self.block3_conv1(f2))
        x = F.relu(self.block3_conv2(x))
        x = F.relu(self.block3_conv3(x))
        pool3 = F.max_pool2d(x, kernel_size=2, stride=2)
        
        # Block 4
        x = F.relu(self.block4_conv1(pool3))
        x = F.relu(self.block4_conv2(x))
        x = F.relu(self.block4_conv3(x))
        pool4 = F.max_pool2d(x, kernel_size=2, stride=2)
        
        # Block 5
        x = F.relu(self.block5_conv1(pool4))
        x = F.relu(self.block5_conv2(x))
        x = F.relu(self.block5_conv3(x))
        pool5 = F.max_pool2d(x, kernel_size=2, stride=2)
        
        # Upsampling for pool4 layer
        conv7_4a = F.relu(self.conv7_4a(pool5))
        conv7_4b = F.relu(self.conv7_4b(conv7_4a))
        conv7_4 = self.conv7_4(conv7_4b)
        
        # Upsampling for pool411
        pool411 = F.relu(self.pool4_11(pool4))
        pool411_b = F.relu(self.pool411_b(pool411))
        pool411_2 = self.pool411_2(pool411_b)
        
        pool311 = F.relu(self.pool3_11(pool3))
        
        # Concatenate and apply add layer
        added = torch.cat((pool411_2, pool311, conv7_4), dim=1)
        added = self.add_layer(added)
        
        # Final upsampling
        output = self.conv8(added)
        output = F.softmax(output, dim=1)
        
        return output
