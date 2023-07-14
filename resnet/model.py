import torch as tc
import torch.nn.functional as F


class IdentityShortcut(tc.nn.Module):
    # identity shortcut connection with automatic zero padding.
    def __init__(self, input_channels, output_channels):
        super(IdentityShortcut, self).__init__()
        assert input_channels <= output_channels
        self.input_channels = input_channels
        self.output_channels = output_channels

    def forward(self, x, r):
        if self.input_channels == self.output_channels:
            return x + r
        else:
            # for NCHW format. pads channels on the right with zeros.
            pad_spec = (0, 0, 0, 0, 0, self.output_channels-self.input_channels)
            return F.pad(x, pad_spec) + r


class DownsamplingIdentityShortcut(tc.nn.Module):
    # identity shortcut connection with automatic zero padding and downsampling when channel size changes.
    def __init__(self, input_channels, output_channels):
        super(DownsamplingIdentityShortcut, self).__init__()
        assert input_channels <= output_channels
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.avgpool = tc.nn.AvgPool2d((2,2), stride=(2,2))

    def forward(self, x, r):
        if self.input_channels == self.output_channels:
            return x + r
        else:
            i = self.avgpool(x)
            # for NCHW format. pads channels on the right with zeros.
            pad_spec = (0, 0, 0, 0, 0, self.output_channels-self.input_channels)
            i = F.pad(i, pad_spec)
            return i + r


class ResBlock(tc.nn.Module):
    def __init__(self, input_channels, output_channels, downsample=False):
        super(ResBlock, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        s = 2 if downsample else 1
        self.conv_sequence = tc.nn.Sequential(
            tc.nn.Conv2d(self.input_channels, self.output_channels, (3,3), stride=(s,s), padding=(1,1), bias=False),
            tc.nn.BatchNorm2d(self.output_channels),
            tc.nn.ReLU(),
            tc.nn.Conv2d(self.output_channels, self.output_channels, (3,3), stride=(1,1), padding=(1,1), bias=False),
            tc.nn.BatchNorm2d(self.output_channels),
        )
        self.identity_shortcut = DownsamplingIdentityShortcut(self.input_channels, self.output_channels)

        for m in self.conv_sequence.modules():
            if isinstance(m, tc.nn.Conv2d):
                tc.nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            elif isinstance(m, tc.nn.BatchNorm2d):
                tc.nn.init.constant_(m.weight, 1.)
                tc.nn.init.constant_(m.bias, 0.)

    def forward(self, x):
        r = self.conv_sequence(x)
        i = self.identity_shortcut(x, r)
        return tc.nn.ReLU()(i)


class PreactivationResBlock(tc.nn.Module):
    # preactivation res net.
    def __init__(self, input_channels, output_channels, downsample=False):
        super(PreactivationResBlock, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        s = 2 if downsample else 1
        self.conv_sequence = tc.nn.Sequential(
            tc.nn.BatchNorm2d(self.input_channels),
            tc.nn.ReLU(),
            tc.nn.Conv2d(self.input_channels, self.output_channels, (3,3), stride=(s,s), padding=(1,1), bias=False),
            tc.nn.BatchNorm2d(self.output_channels),
            tc.nn.ReLU(),
            tc.nn.Conv2d(self.output_channels, self.output_channels, (3,3), stride=(1,1), padding=(1,1), bias=False)
        )
        self.identity_shortcut = DownsamplingIdentityShortcut(self.input_channels, self.output_channels)

        for m in self.conv_sequence.modules():
            if isinstance(m, tc.nn.Conv2d):
                tc.nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            elif isinstance(m, tc.nn.BatchNorm2d):
                tc.nn.init.constant_(m.weight, 1.)
                tc.nn.init.constant_(m.bias, 0.)

    def forward(self, x):
        r = self.conv_sequence(x)
        i = self.identity_shortcut(x, r)
        return i


class GlobalAveragePool2D(tc.nn.Module):
    def __init_(self):
        super(GlobalAveragePool2D, self).__init__()

    def forward(self, x):
        return x.mean(dim=(2,3)) # assumes N, C, H, W format.


class InitialResNetConv(tc.nn.Module):
    def __init__(self, img_channels, initial_num_filters):
        super(InitialResNetConv, self).__init__()
        self.img_channels = img_channels
        self.initial_num_filters = initial_num_filters

        self.conv_sequence = tc.nn.Sequential(
            tc.nn.Conv2d(self.img_channels, self.initial_num_filters, (3,3), stride=(1,1), padding=(1,1), bias=False),
            tc.nn.BatchNorm2d(self.initial_num_filters, momentum=0.9, track_running_stats=True),
            tc.nn.ReLU()
        )

        for m in self.conv_sequence.modules():
            if isinstance(m, tc.nn.Conv2d):
                tc.nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            elif isinstance(m, tc.nn.BatchNorm2d):
                tc.nn.init.constant_(m.weight, 1.)
                tc.nn.init.constant_(m.bias, 0.)

    def forward(self, x):
        h = self.conv_sequence(x)
        return h


class Cifar10ResNet(tc.nn.Module):
    # From section 4.2 of He et al., 2015 - 'Deep Residual Learning for Image Recognition'.
    # This differs from the ImageNet resnets due to the initial convolution kernel size,
    # as well as the number of feature maps, which start at 16 rather than 64.
    def __init__(self, img_height, img_width, img_channels, initial_num_filters, num_classes, num_repeats, num_stages):
        super(Cifar10ResNet, self).__init__()
        self.img_height = img_height
        self.img_width = img_width
        self.img_channels = img_channels
        self.initial_num_filters = initial_num_filters
        self.num_classes = num_classes

        self.num_repeats = num_repeats
        self.num_stages = num_stages

        self.initial_conv = InitialResNetConv(img_channels, initial_num_filters)
        self.blocks = tc.nn.ModuleList()

        for s in range(self.num_stages):
            for i in range(self.num_repeats):
                if s == 0:
                    fin = self.initial_num_filters
                    fout = self.initial_num_filters
                    self.blocks.append(ResBlock(fin, fout))
                elif s != 0 and i == 0:
                    fin = self.initial_num_filters * (2 ** (s-1))
                    fout = self.initial_num_filters * (2 ** s)
                    self.blocks.append(ResBlock(fin, fout, downsample=True))
                else:
                    fin = self.initial_num_filters * (2 ** s)
                    fout = self.initial_num_filters * (2 ** s)
                    self.blocks.append(ResBlock(fin, fout))

        self.avgpool = GlobalAveragePool2D()
        self.final_num_filters = self.initial_num_filters * (2 ** (self.num_stages-1))
        self.fc = tc.nn.Linear(self.final_num_filters, self.num_classes, bias=False)
        tc.nn.init.kaiming_normal(self.fc.weight)

    def forward(self, x):
        x = self.initial_conv(x)
        for l in range(self.num_stages * self.num_repeats):
            x = self.blocks[l](x)
        spatial_features = x
        pooled_features = self.avgpool(spatial_features)
        logits = self.fc(pooled_features)
        return logits

    def visualize(self, x):
        x = self.initial_conv(x)
        for l in range(self.num_stages * self.num_repeats):
            x = self.blocks[l](x)
        spatial_features = x

        target_shape = (-1, self.final_num_filters)
        spatial_features_batched = tc.reshape(spatial_features, target_shape)
        logits = self.fc(spatial_features_batched)
        argmax_logits = tc.argmax(logits, dim=1)

        spatial_shape = (-1, (self.img_height // (2 ** (self.num_stages-1))), (self.img_width // (2 ** (self.num_stages-1))))
        argmax_logits = tc.reshape(argmax_logits, spatial_shape)

        return argmax_logits


class Cifar10PreactivationResNet(tc.nn.Module):
    # Preactivation resnet variant of the Cifar-10 resnet above.
    # Preactivation resnets are described in He et al., 2016 - 'Identity Mappings in Deep Residual Networks'
    def __init__(self, img_height, img_width, img_channels, initial_num_filters, num_classes, num_repeats, num_stages):
        super(Cifar10PreactivationResNet, self).__init__()
        self.img_height = img_height
        self.img_width = img_width
        self.img_channels = img_channels
        self.initial_num_filters = initial_num_filters
        self.num_classes = num_classes

        self.num_repeats = num_repeats
        self.num_stages = num_stages

        self.initial_conv = InitialResNetConv(img_channels, initial_num_filters)
        self.blocks = tc.nn.ModuleList()

        for s in range(self.num_stages):
            for i in range(self.num_repeats):
                if s == 0:
                    fin = self.initial_num_filters
                    fout = self.initial_num_filters
                    self.blocks.append(PreactivationResBlock(fin, fout))
                elif s != 0 and i == 0:
                    fin = self.initial_num_filters * (2 ** (s-1))
                    fout = self.initial_num_filters * (2 ** s)
                    self.blocks.append(PreactivationResBlock(fin, fout, downsample=True))
                else:
                    fin = self.initial_num_filters * (2 ** s)
                    fout = self.initial_num_filters * (2 ** s)
                    self.blocks.append(PreactivationResBlock(fin, fout))

        self.avgpool = GlobalAveragePool2D()
        self.final_num_filters = self.initial_num_filters * (2 ** (self.num_stages-1))
        self.fc = tc.nn.Linear(self.final_num_filters, self.num_classes, bias=False)
        tc.nn.init.kaiming_normal(self.fc.weight)

    def forward(self, x):
        x = self.initial_conv(x)
        for l in range(self.num_stages * self.num_repeats):
            x = self.blocks[l](x)
        spatial_features = x
        pooled_features = self.avgpool(spatial_features)
        logits = self.fc(pooled_features)
        return logits

    def visualize(self, x):
        x = self.initial_conv(x)
        for l in range(self.num_stages * self.num_repeats):
            x = self.blocks[l](x)
        spatial_features = x

        target_shape = (-1, self.final_num_filters)
        spatial_features_batched = tc.reshape(spatial_features, target_shape)
        logits = self.fc(spatial_features_batched)
        argmax_logits = tc.argmax(logits, dim=1)

        spatial_shape = (-1, (self.img_height // (2 ** (self.num_stages-1))), (self.img_width // (2 ** (self.num_stages-1))))
        argmax_logits = tc.reshape(argmax_logits, spatial_shape)

        return argmax_logits

