import torch
import torch.nn as nn
import re


# 1. 複製剛剛 fast_style_transfer.py 裡面的網路架構 (TransformerNet 及相關元件)
# (為了保持程式碼獨立可執行，這裡再次定義網路)
class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        reflection_padding = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in1 = nn.InstanceNorm2d(channels, affine=True)
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in2 = nn.InstanceNorm2d(channels, affine=True)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.in1(self.conv1(x)))
        out = self.in2(self.conv2(out))
        out = out + residual
        return out


class UpsampleConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
        super(UpsampleConvLayer, self).__init__()
        self.upsample = upsample
        reflection_padding = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        x_in = x
        if self.upsample:
            x_in = torch.nn.functional.interpolate(
                x_in, mode="nearest", scale_factor=self.upsample
            )
        out = self.reflection_pad(x_in)
        out = self.conv2d(out)
        return out


class TransformerNet(nn.Module):
    def __init__(self):
        super(TransformerNet, self).__init__()
        self.conv1 = ConvLayer(3, 32, kernel_size=9, stride=1)
        self.in1 = nn.InstanceNorm2d(32, affine=True)
        self.conv2 = ConvLayer(32, 64, kernel_size=3, stride=2)
        self.in2 = nn.InstanceNorm2d(64, affine=True)
        self.conv3 = ConvLayer(64, 128, kernel_size=3, stride=2)
        self.in3 = nn.InstanceNorm2d(128, affine=True)
        self.res1 = ResidualBlock(128)
        self.res2 = ResidualBlock(128)
        self.res3 = ResidualBlock(128)
        self.res4 = ResidualBlock(128)
        self.res5 = ResidualBlock(128)
        self.deconv1 = UpsampleConvLayer(128, 64, kernel_size=3, stride=1, upsample=2)
        self.in4 = nn.InstanceNorm2d(64, affine=True)
        self.deconv2 = UpsampleConvLayer(64, 32, kernel_size=3, stride=1, upsample=2)
        self.in5 = nn.InstanceNorm2d(32, affine=True)
        self.deconv3 = ConvLayer(32, 3, kernel_size=9, stride=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.relu(self.in1(self.conv1(x)))
        out = self.relu(self.in2(self.conv2(out)))
        out = self.relu(self.in3(self.conv3(out)))
        out = self.res1(out)
        out = self.res2(out)
        out = self.res3(out)
        out = self.res4(out)
        out = self.res5(out)
        out = self.relu(self.in4(self.deconv1(out)))
        out = self.relu(self.in5(self.deconv2(out)))
        out = self.deconv3(out)
        return out


# 2. 載入權重並準備匯出
print("正在載入 PyTorch 模型...")
model = TransformerNet()
state_dict = torch.load(
    "mosaic.pth", map_location="cpu"
)  # 匯出 ONNX 時在 CPU 上操作即可
for k in list(state_dict.keys()):
    if re.search(r"in\d+\.running_(mean|var)$", k):
        del state_dict[k]
model.load_state_dict(state_dict, strict=False)
model.eval()

# 3. 建立一個虛擬輸入 (Dummy Input)
# 假設講堂攝影機拉流後，我們前處理會將影片 resize 到 480x640 來做推論
# Batch Size = 1, Channels = 3, Height = 480, Width = 640
dummy_input = torch.randn(1, 3, 480, 640)

# 4. 匯出 ONNX
onnx_filename = "mosaic_480x640.onnx"
print(f"正在匯出為 {onnx_filename} ...")

torch.onnx.export(
    model,
    dummy_input,
    onnx_filename,
    export_params=True,
    opset_version=11,  # TensorRT 通常對 opset 11 支援度最好
    do_constant_folding=True,  # 執行常數摺疊優化
    input_names=["input"],  # 指定輸入節點名稱
    output_names=["output"],  # 指定輸出節點名稱
)

print("ONNX 模型匯出成功！")
