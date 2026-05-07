import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import os
from io import BytesIO
import time

os.environ["TORCH_HOME"] = "./weights"
# 判斷硬體資源 (有 GPU 就用 GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"目前使用的運算設備: {device}")


# ==========================================
# 1. 圖片讀取與前處理 (改為讀取本地端檔案)
# ==========================================
def load_image(image_path, size=None, max_size=400):
    image = Image.open(image_path).convert("RGB")

    if size is not None:
        # 如果有給定 size，強制縮放至指定的 (H, W)
        transform = transforms.Compose(
            [
                transforms.Resize(size),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )
    else:
        # 等比例縮放，控制最大邊
        s = max_size if max(image.size) > max_size else max(image.size)
        transform = transforms.Compose(
            [
                transforms.Resize(s),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

    image = transform(image).unsqueeze(0)
    return image.to(device)


def imsave(tensor, filename):
    image = tensor.cpu().clone().squeeze(0)
    # 反正規化
    image = image * torch.tensor((0.229, 0.224, 0.225)).view(3, 1, 1) + torch.tensor(
        (0.485, 0.456, 0.406)
    ).view(3, 1, 1)
    image = image.clamp(0, 1)
    transform = transforms.ToPILImage()
    image = transform(image)
    image.save(filename)


print("正在讀取本地端測試圖片...")
# 讀取內容圖
content_img = load_image("content.png", max_size=400)
# 取得內容圖的長寬，並強制讓風格圖跟著這個尺寸走
target_size = (content_img.shape[2], content_img.shape[3])
style_img = load_image("style.png", size=target_size)

# 初始輸入圖片設定為內容圖片的複製，並開啟梯度追蹤
input_img = content_img.clone().requires_grad_(True)


# ==========================================
# 2. 建立格拉姆矩陣與損失函數 (對應簡報公式)
# ==========================================
def gram_matrix(input):
    a, b, c, d = input.size()  # a=batch, b=特徵圖數量(channel), c,d=長寬
    features = input.view(a * b, c * d)
    G = torch.mm(features, features.t())  # 內積計算共現關聯度
    return G.div(a * b * c * d)


class ContentLoss(nn.Module):
    def __init__(self, target):
        super(ContentLoss, self).__init__()
        self.target = target.detach()

    def forward(self, input):
        self.loss = nn.functional.mse_loss(input, self.target)
        return input


class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = nn.functional.mse_loss(G, self.target)
        return input


# ==========================================
# 3. 載入預訓練的 VGG19 模型
# ==========================================
print("正在載入 VGG19 模型...")
cnn = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features.to(device).eval()

# 設定特徵萃取的層數 (對應簡報中 conv4_2, conv1_1 等設定)
content_layers = ["conv_4"]
style_layers = ["conv_1", "conv_2", "conv_3", "conv_4", "conv_5"]

content_losses = []
style_losses = []
model = nn.Sequential()

i = 0
for layer in cnn.children():
    if isinstance(layer, nn.Conv2d):
        i += 1
        name = f"conv_{i}"
    elif isinstance(layer, nn.ReLU):
        name = f"relu_{i}"
        layer = nn.ReLU(inplace=False)
    elif isinstance(layer, nn.MaxPool2d):
        name = f"pool_{i}"
    elif isinstance(layer, nn.BatchNorm2d):
        name = f"bn_{i}"
    else:
        raise RuntimeError(f"Unrecognized layer: {layer.__class__.__name__}")

    model.add_module(name, layer)

    if name in content_layers:
        target = model(content_img).detach()
        content_loss = ContentLoss(target)
        model.add_module(f"content_loss_{i}", content_loss)
        content_losses.append(content_loss)

    if name in style_layers:
        target_feature = model(style_img).detach()
        style_loss = StyleLoss(target_feature)
        model.add_module(f"style_loss_{i}", style_loss)
        style_losses.append(style_loss)

# 裁切掉不需要計算的後方網路層以節省資源
for i in range(len(model) - 1, -1, -1):
    if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
        break
model = model[: (i + 1)]

# ==========================================
# 4. 開始執行風格轉換優化
# ==========================================
print("開始進行神經風格轉換 (Gatys 迭代法)...")
optimizer = optim.LBFGS([input_img])

num_steps = 300  # 實務上通常要 300 步以上，這裡設 100 步先看效果
style_weight = 100000  # 稍微調低為 1e5，避免初期梯度過大
content_weight = 1

run = [0]
start_time = time.time()

while run[0] <= num_steps:

    def closure():
        # 移除原本這裡的 clamp_(0, 1)，因為張量目前是正規化狀態
        optimizer.zero_grad()
        model(input_img)

        style_score = 0
        content_score = 0

        for sl in style_losses:
            style_score += sl.loss
        for cl in content_losses:
            content_score += cl.loss

        style_score *= style_weight
        content_score *= content_weight

        loss = style_score + content_score
        loss.backward()

        run[0] += 1
        if run[0] % 20 == 0:
            print(
                f"迭代 {run[0]}/{num_steps} | Style Loss: {style_score.item():.4f} Content Loss: {content_score.item():.4f}"
            )
        return style_score + content_score

    optimizer.step(closure)

end_time = time.time()

print(f"轉換完成！總耗時: {end_time - start_time:.2f} 秒")
imsave(input_img, "output_baseline.jpg")
print("結果已存檔為 output_baseline.jpg")
