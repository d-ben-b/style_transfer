import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import urllib.request
import os
import time
import re

# 判斷硬體資源
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"目前使用的運算設備: {device}")


# ==========================================
# 1. 定義影像轉換網路 (Image Transform Net)
# 這是基於 Johnson 等人論文的標準架構
# ==========================================
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


# ==========================================
# 2. 下載權重與讀取圖片
# ==========================================
# 自動下載預訓練好的馬賽克風格權重
if __name__ == "__main__":
    weight_file = "mosaic.pth"
    if not os.path.exists(weight_file):
        print("正在下載預訓練的馬賽克風格權重...")
        url = "https://raw.githubusercontent.com/vihar/picasso-style-transfer/master/saved_models/mosaic.pth"
        urllib.request.urlretrieve(url, weight_file)

    print("正在載入模型與權重...")
    style_model = TransformerNet()
    state_dict = torch.load(weight_file)
    # 修正一下歷史版本模型儲存的 key 名稱差異
    for k in list(state_dict.keys()):
        if re.search(r"in\d+\.running_(mean|var)$", k):
            del state_dict[k]
    style_model.load_state_dict(state_dict, strict=False)
    style_model.to(device)
    style_model.eval()  # 設定為推論模式

    print("正在讀取本地端測試圖片 content.png ...")
    content_image = Image.open("content.png").convert("RGB")
    content_transform = transforms.Compose(
        [
            transforms.Resize(
                480
            ),  # 將圖片短邊縮放至 480 像素 (若還是 OOM 可以改成 256)
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.mul(255)),
        ]
    )
    content_tensor = content_transform(content_image).unsqueeze(0).to(device)

    # ==========================================
    # 3. 執行推論 (Forward Pass)
    # ==========================================
    print("開始進行 Fast Neural Style Transfer 推論...")
    start_time = time.time()

    with torch.no_grad():  # 推論時不需要計算梯度
        output_tensor = style_model(content_tensor)

    end_time = time.time()
    print(f"轉換完成！總耗時: {end_time - start_time:.4f} 秒")

    # ==========================================
    # 4. 儲存結果
    # ==========================================
    output_tensor = output_tensor.clone().squeeze(0).cpu().clamp(0, 255)
    output_tensor = output_tensor.div(255)
    imsave_transform = transforms.ToPILImage()
    output_image = imsave_transform(output_tensor)
    output_image.save("output_fast_baseline.png")
    print("結果已存檔為 output_fast_baseline.png")
