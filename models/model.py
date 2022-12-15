# Neural network model
import torch
import torch.nn as nn
from models.optimizer import load_optimizer, load_scheduler
from utils.util import check_previous_training
from torchvision import models
import torchvision
import os.path as osp


def load_best_model(opt, checkpoint_directory):
    best_model_path = osp.join(checkpoint_directory, 'best_model_state_dict.pth')
    checkpoint = torch.load(best_model_path)
    model = load_model(opt, '')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.cuda()
    model.eval()
    return model


def load_model(opt, checkpoint_directory):
    # cont_train input as True or False
    # Load and return model, optimizer, scheduler
    # Load model from directory

    print(f'Loading model {str(opt.model)}')
    if opt.model == 'UNet':
        model = UNeuralNetwork(opt.in_channels, opt.out_channels)
    if opt.model == 'UNetWithResnet34':
        model = UNetWithResnet34(opt.in_channels, opt.out_channels)
    if opt.model == 'SegNet':
        model = SegNet(opt.out_channels)

    # Return model for testing
    if checkpoint_directory == '':
        print(f'Loading model {opt.model}...')
        return model
        
    if torch.cuda.is_available():
        model.cuda()
        
    opt = check_previous_training(opt)

    optimizer = load_optimizer(opt, model)
    scheduler = load_scheduler(opt, optimizer)

    if opt.cont_train:
        model_path = osp.join(checkpoint_directory, 'model_state_dict.pth')
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load options, model, optimizer, scheduler if model have not finish training
        opt.last_epoch = checkpoint['last_epoch']
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    

    return opt, model, optimizer, scheduler


class UNeuralNetwork(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = self.DownBlock(in_channels, 16, 3, 'same')
        self.conv2 = self.DownBlock(16, 32, 3, 'same')
        self.conv3 = self.DownBlock(32, 64, 3, 'same')
        self.conv4 = self.DownBlock(64, 128, 3, 'same')

        self.upconv1 = self.UpBlock(128, 64, 3, 'same')
        self.upconv2 = self.UpBlock(64 * 2, 32, 3, 'same')
        self.upconv3 = self.UpBlock(32 * 2, 16, 3, 'same')
        self.upconv4 = self.UpBlock(16 * 2, out_channels, 3, 'same')

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)

        upconv1 = self.upconv1(conv4)
        upconv2 = self.upconv2(torch.cat([upconv1, conv3], 1))
        upconv3 = self.upconv3(torch.cat([upconv2, conv2], 1))
        upconv4 = self.upconv4(torch.cat([upconv3, conv1], 1))

        logits = upconv4
        return logits

    def DownBlock(self, in_channels, out_channels, kernel_size, padding):
        down_model = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        return down_model

    def UpBlock(self, in_channels, out_channels, kernel_size, padding):
        up_model = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
        )

        return up_model


class SegNet(nn.Module):
    def __init__(self, num_classes, in_channels=3, pretrained=True, freeze_bn=False, **_):
        super(SegNet, self).__init__()
        vgg_bn = models.vgg16_bn(pretrained=pretrained)
        encoder = list(vgg_bn.features.children())

        if in_channels != 3:
            encoder[0] = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding='same')

        # Encoder, VGG without any maxpooling
        self.stage1_encoder = nn.Sequential(*encoder[:6])
        self.stage2_encoder = nn.Sequential(*encoder[7:13])
        self.stage3_encoder = nn.Sequential(*encoder[14:23])
        self.stage4_encoder = nn.Sequential(*encoder[24:33])
        self.stage5_encoder = nn.Sequential(*encoder[34:-1])
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

        # TO MODIFY
        # Decoder trainable
        decoder = encoder
        decoder = [i for i in list(reversed(decoder)) if not isinstance(i, nn.MaxPool2d)]
        # Replace the last conv layer
        decoder[-1] = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding='same')
        # When reversing, we also reversed conv->batchN->relu, correct it
        decoder = [item for i in range(0, len(decoder), 3) for item in decoder[i:i + 3][::-1]]
        # Replace some conv layers & batchN after them
        for i, module in enumerate(decoder):
            if isinstance(module, nn.Conv2d):
                if module.in_channels != module.out_channels:
                    decoder[i + 1] = nn.BatchNorm2d(module.in_channels)
                    decoder[i] = nn.Conv2d(module.out_channels, module.in_channels, kernel_size=3, stride=1, padding='same')

        self.stage1_decoder = nn.Sequential(*decoder[0:9])
        self.stage2_decoder = nn.Sequential(*decoder[9:18])
        self.stage3_decoder = nn.Sequential(*decoder[18:27])
        self.stage4_decoder = nn.Sequential(*decoder[27:33])
        self.stage5_decoder = nn.Sequential(*decoder[33:],
                                            nn.Conv2d(64, num_classes, kernel_size=3, stride=1, padding='same')
                                            )
        self.unpool = nn.MaxUnpool2d(kernel_size=2, stride=2)

        self._initialize_weights(self.stage1_decoder, self.stage2_decoder, self.stage3_decoder,
                                 self.stage4_decoder, self.stage5_decoder)
        if freeze_bn:
            self.freeze_bn()

    def _initialize_weights(self, *stages):
        for modules in stages:
            for module in modules.modules():
                if isinstance(module, nn.Conv2d):
                    nn.init.kaiming_normal_(module.weight)
                    if module.bias is not None:
                        module.bias.data.zero_()
                elif isinstance(module, nn.BatchNorm2d):
                    module.weight.data.fill_(1)
                    module.bias.data.zero_()

    def forward(self, x):
        # Encoder
        x = self.stage1_encoder(x)
        x1_size = x.size()
        x, indices1 = self.pool(x)

        x = self.stage2_encoder(x)
        x2_size = x.size()
        x, indices2 = self.pool(x)

        x = self.stage3_encoder(x)
        x3_size = x.size()
        x, indices3 = self.pool(x)

        x = self.stage4_encoder(x)
        x4_size = x.size()
        x, indices4 = self.pool(x)

        x = self.stage5_encoder(x)
        x5_size = x.size()
        x, indices5 = self.pool(x)

        # Decoder
        x = self.unpool(x, indices=indices5, output_size=x5_size)
        x = self.stage1_decoder(x)

        x = self.unpool(x, indices=indices4, output_size=x4_size)
        x = self.stage2_decoder(x)

        x = self.unpool(x, indices=indices3, output_size=x3_size)
        x = self.stage3_decoder(x)

        x = self.unpool(x, indices=indices2, output_size=x2_size)
        x = self.stage4_decoder(x)

        x = self.unpool(x, indices=indices1, output_size=x1_size)
        x = self.stage5_decoder(x)

        return x

    def get_backbone_params(self):
        return []

    def get_decoder_params(self):
        return self.parameters()

    def freeze_bn(self):
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()


class UNetWithResnet34(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        resnet34 = torchvision.models.resnet.resnet34(pretrained=True)
        down_blocks = []
        for bottleneck in list(resnet34.children()):
            if isinstance(bottleneck, nn.Sequential):
                down_blocks.append(bottleneck)

        # Encoder
        self.conv1 = nn.Sequential(*list(resnet34.children()))[:3]
        self.input_pool = list(resnet34.children())[3]
        self.conv2 = down_blocks[0]
        self.conv3 = down_blocks[1]
        self.conv4 = down_blocks[2]
        self.conv5 = down_blocks[3]

        # Copy and concat for decoder
        self.copy_conv1 = self.copy_conv(64, 128, 1, 1)
        self.copy_conv2 = self.copy_conv(64, 128, 1, 1)
        self.copy_conv3 = self.copy_conv(128, 128, 1, 1)
        self.copy_conv4 = self.copy_conv(256, 128, 1, 1)

        # Decoder
        self.upconv1 = self.UpBlock(512, 128, 2, 'same')
        self.upconv2 = self.UpBlockWithSCSE(128, 128, 2, 'same')
        self.upconv3 = self.UpBlockWithSCSE(128, 128, 2, 'same')
        self.upconv4 = self.UpBlockWithSCSE(128, 128, 2, 'same')
        self.upconv5 = self.UpBlockWithSCSE(128, out_channels, 2, 'same')

    def forward(self, x):
        conv1 = self.conv1(x)
        copy_conv1 = self.copy_conv1(conv1)
        conv1 = self.input_pool(conv1)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)
        copy_conv2 = self.copy_conv2(conv2)
        copy_conv3 = self.copy_conv3(conv3)
        copy_conv4 = self.copy_conv4(conv4)

        upconv1 = self.upconv1(conv5)
        upconv2 = self.upconv2(torch.cat([upconv1, copy_conv4], 1))
        upconv3 = self.upconv3(torch.cat([upconv2, copy_conv3], 1))
        upconv4 = self.upconv4(torch.cat([upconv3, copy_conv2], 1))
        upconv5 = self.upconv4(torch.cat([upconv4, copy_conv1], 1))

        logits = upconv5
        return logits

    def DownBlock(self, in_channels, out_channels, kernel_size, padding):
        down_model = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        return down_model

    def UpBlock(self, in_channels, out_channels, kernel_size, padding, reduction_ratio=0.5):
        up_model = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.ConvTranspose2d(out_channels, out_channels, kernel_size=kernel_size, stride=2, padding=1, output_padding=1)
        )

        return up_model
    
    def UpBlockWithSCSE(self, in_channels, out_channels, kernel_size, padding, reduction_ratio=0.5):
        up_model = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            ChannelSpatialSELayer(out_channels, reduction_ratio=reduction_ratio),
            nn.ConvTranspose2d(int(out_channels/reduction_ratio), out_channels, kernel_size=kernel_size, stride=2, padding=1, output_padding=1)
        )

        return up_model

    def copy_conv(self, in_channels, out_channels, kernel_size, padding):
        copy = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding)

        return copy


class ChannelSELayer(nn.Module):
    """
    Re-implementation of Squeeze-and-Excitation (SE) block described in:
        *Hu et al., Squeeze-and-Excitation Networks, arXiv:1709.01507*
    """

    def __init__(self, num_channels, reduction_ratio=2):
        """
        :param num_channels: No of input channels
        :param reduction_ratio: By how much should the num_channels should be reduced
        """
        super(ChannelSELayer, self).__init__()
        num_channels_reduced = int(num_channels // reduction_ratio)
        self.reduction_ratio = reduction_ratio
        self.fc1 = nn.Linear(num_channels, num_channels_reduced, bias=True)
        self.fc2 = nn.Linear(num_channels_reduced, num_channels, bias=True)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor):
        """
        :param input_tensor: X, shape = (batch_size, num_channels, H, W)
        :return: output tensor
        """
        batch_size, num_channels, H, W = input_tensor.size()
        # Average along each channel
        squeeze_tensor = input_tensor.view(batch_size, num_channels, -1).mean(dim=2)

        # channel excitation
        fc_out_1 = self.relu(self.fc1(squeeze_tensor))
        fc_out_2 = self.sigmoid(self.fc2(fc_out_1))

        a, b = squeeze_tensor.size()
        output_tensor = torch.mul(input_tensor, fc_out_2.view(a, b, 1, 1))
        return output_tensor


class SpatialSELayer(nn.Module):
    """
    Re-implementation of SE block -- squeezing spatially and exciting channel-wise described in:
        *Roy et al., Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Networks, MICCAI 2018*
    """

    def __init__(self, num_channels):
        """
        :param num_channels: No of input channels
        """
        super(SpatialSELayer, self).__init__()
        self.conv = nn.Conv2d(num_channels, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor, weights=None):
        """
        :param weights: weights for few shot learning
        :param input_tensor: X, shape = (batch_size, num_channels, H, W)
        :return: output_tensor
        """
        # spatial squeeze
        batch_size, channel, a, b = input_tensor.size()

        if weights is not None:
            weights = torch.mean(weights, dim=0)
            weights = weights.view(1, channel, 1, 1)
            out = F.conv2d(input_tensor, weights)
        else:
            out = self.conv(input_tensor)
        squeeze_tensor = self.sigmoid(out)

        # spatial excitation
        # print(input_tensor.size(), squeeze_tensor.size())
        squeeze_tensor = squeeze_tensor.view(batch_size, 1, a, b)
        output_tensor = torch.mul(input_tensor, squeeze_tensor)
        # output_tensor = torch.mul(input_tensor, squeeze_tensor)
        return output_tensor


class ChannelSpatialSELayer(nn.Module):
    """
    Re-implementation of concurrent spatial and channel squeeze & excitation:
        *Roy et al., Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Networks, MICCAI 2018, arXiv:1803.02579*
    """

    def __init__(self, num_channels, reduction_ratio=2):
        """
        :param num_channels: No of input channels
        :param reduction_ratio: By how much should the num_channels should be reduced
        """
        super(ChannelSpatialSELayer, self).__init__()
        self.cSE = ChannelSELayer(num_channels, reduction_ratio)
        self.sSE = SpatialSELayer(num_channels)

    def forward(self, input_tensor):
        """
        :param input_tensor: X, shape = (batch_size, num_channels, H, W)
        :return: output_tensor
        """
        output_tensor = torch.max(self.cSE(input_tensor), self.sSE(input_tensor))
        return output_tensor
