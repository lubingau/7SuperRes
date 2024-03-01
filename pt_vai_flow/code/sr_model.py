import torch
import torch.nn as nn

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