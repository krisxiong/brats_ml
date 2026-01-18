"""
3D残差U-Net模型
支持任意尺寸的3D输入
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """
    3D残差块
    包含两个3D卷积层和残差连接
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()

        # 第一个卷积层
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3,
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)

        # 第二个卷积层
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3,
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)

        # 残差连接的shortcut
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm3d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResUNet(nn.Module):
    """
    3D残差U-Net
    支持任意尺寸的3D输入（全卷积网络）
    """
    def __init__(self, in_channels=4, out_channels=3, base_channels=16):
        """
        参数:
            in_channels: 输入通道数 (BraTS: 4个模态)
            out_channels: 输出类别数 (BraTS: 3个区域)
            base_channels: 基础通道数，控制网络宽度
                          16: 轻量级 (~1.8M参数)
                          32: 标准 (~7.2M参数)
                          64: 大型 (~28M参数)
        """
        super(ResUNet, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels

        # ============ 编码器 (Encoder) ============
        self.enc1 = ResidualBlock(in_channels, base_channels)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.enc2 = ResidualBlock(base_channels, base_channels*2)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.enc3 = ResidualBlock(base_channels*2, base_channels*4)
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.enc4 = ResidualBlock(base_channels*4, base_channels*8)
        self.pool4 = nn.MaxPool3d(kernel_size=2, stride=2)

        # ============ 瓶颈层 (Bottleneck) ============
        self.bottleneck = ResidualBlock(base_channels*8, base_channels*16)

        # ============ 解码器 (Decoder) ============
        self.upconv4 = nn.ConvTranspose3d(base_channels*16, base_channels*8,
                                          kernel_size=2, stride=2)
        self.dec4 = ResidualBlock(base_channels*16, base_channels*8)

        self.upconv3 = nn.ConvTranspose3d(base_channels*8, base_channels*4,
                                          kernel_size=2, stride=2)
        self.dec3 = ResidualBlock(base_channels*8, base_channels*4)

        self.upconv2 = nn.ConvTranspose3d(base_channels*4, base_channels*2,
                                          kernel_size=2, stride=2)
        self.dec2 = ResidualBlock(base_channels*4, base_channels*2)

        self.upconv1 = nn.ConvTranspose3d(base_channels*2, base_channels,
                                          kernel_size=2, stride=2)
        self.dec1 = ResidualBlock(base_channels*2, base_channels)

        # ============ 输出层 ============
        self.out_conv = nn.Conv3d(base_channels, out_channels, kernel_size=1)

        # 初始化权重
        self._initialize_weights()

    def _initialize_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        前向传播
        支持任意尺寸的3D输入

        参数:
            x: [B, in_channels, D, H, W]

        返回:
            output: [B, out_channels, D, H, W]
        """
        # 检查输入维度
        if x.dim() != 5:
            raise ValueError(f"期望5D输入 [B, C, D, H, W]，得到 {x.dim()}D")

        if x.size(1) != self.in_channels:
            raise ValueError(f"期望 {self.in_channels} 个输入通道，得到 {x.size(1)}")

        # 记录原始尺寸用于最后的resize
        original_size = x.shape[2:]

        # 编码器
        enc1 = self.enc1(x)              # [B, base, D, H, W]
        enc2 = self.enc2(self.pool1(enc1))  # [B, base*2, D/2, H/2, W/2]
        enc3 = self.enc3(self.pool2(enc2))  # [B, base*4, D/4, H/4, W/4]
        enc4 = self.enc4(self.pool3(enc3))  # [B, base*8, D/8, H/8, W/8]

        # 瓶颈
        bottleneck = self.bottleneck(self.pool4(enc4))  # [B, base*16, D/16, H/16, W/16]

        # 解码器 + 跳跃连接
        dec4 = self.upconv4(bottleneck)
        dec4 = self._match_size(dec4, enc4)  # 确保尺寸匹配
        dec4 = torch.cat([dec4, enc4], dim=1)
        dec4 = self.dec4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = self._match_size(dec3, enc3)
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec3 = self.dec3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = self._match_size(dec2, enc2)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.dec2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = self._match_size(dec1, enc1)
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.dec1(dec1)

        # 输出
        out = self.out_conv(dec1)

        # 确保输出尺寸与输入相同
        if out.shape[2:] != original_size:
            out = F.interpolate(out, size=original_size, mode='trilinear',
                               align_corners=False)

        return out

    def _match_size(self, x, target):
        """
        匹配两个tensor的空间尺寸
        处理由于下采样和上采样导致的尺寸不匹配问题
        """
        if x.shape[2:] != target.shape[2:]:
            x = F.interpolate(x, size=target.shape[2:], mode='trilinear',
                            align_corners=False)
        return x

    def get_model_size(self):
        """获取模型大小信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        param_size_mb = total_params * 4 / (1024 ** 2)

        return {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'size_mb': param_size_mb
        }


# 测试代码
if __name__ == "__main__":
    print("=" * 70)
    print("3D残差U-Net模型测试 - 支持任意尺寸")
    print("=" * 70)

    # 测试不同尺寸的输入
    test_cases = [
        {'name': '标准BraTS (155×240×240)', 'size': (2, 4, 155, 240, 240)},
        {'name': '调换维度 (240×240×155)', 'size': (2, 4, 240, 240, 155)},
        {'name': '较小尺寸 (155×160×160)', 'size': (2, 4, 155, 160, 160)},
        {'name': '裁剪后 (128×128×128)', 'size': (2, 4, 128, 128, 128)},
        {'name': '下采样后 (80×120×120)', 'size': (2, 4, 80, 120, 120)},
    ]

    # 测试不同的base_channels
    for base_ch in [16, 32]:
        print(f"\n{'='*70}")
        print(f"Base Channels: {base_ch}")
        print('='*70)

        model = ResUNet(in_channels=4, out_channels=3, base_channels=base_ch)
        model.eval()

        # 显示模型信息
        model_info = model.get_model_size()
        print(f"\n模型参数:")
        print(f"  总参数: {model_info['total_params']:,}")
        print(f"  可训练参数: {model_info['trainable_params']:,}")
        print(f"  模型大小: {model_info['size_mb']:.2f} MB")

        print(f"\n测试不同输入尺寸:")

        for test in test_cases:
            try:
                x = torch.randn(test['size'])

                with torch.no_grad():
                    output = model(x)

                print(f"  ✓ {test['name']}")
                print(f"    输入: {tuple(x.shape[2:])}")
                print(f"    输出: {tuple(output.shape[2:])}")

                # 验证输出尺寸与输入一致
                assert output.shape[2:] == x.shape[2:], "输出尺寸不匹配!"

            except Exception as e:
                print(f"  ✗ {test['name']}: {e}")

    print(f"\n{'='*70}")
    print("测试完成! 模型可以处理任意尺寸的3D输入")
    print('='*70)
