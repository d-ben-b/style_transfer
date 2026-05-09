# 邊緣運算裝置影像風格轉換效能優化實驗報告

## 1. 實驗摘要 (Abstract)
本實驗旨在研究將神經風格轉換（Neural Style Transfer, NST）模型部署於邊緣運算裝置 NVIDIA Jetson Orin Nano 之效能優化策略。透過 TensorRT 框架進行 FP16 與 INT8 量化壓縮，探討模型推論速度（FPS）在不同功耗模式與量化精準度下的表現。實驗結果顯示，在 Orin Nano 預設模式下，FP16 結合非同步推論可達到 21.31 FPS，而 INT8 量化受限於記憶體頻寬與格式轉換開銷，效能提升未如預期，甚至在 MAXN 模式下因硬體降頻而導致效能下降。

## 2. 實驗環境 (Experimental Environment)
* **硬體平台**: NVIDIA Jetson Orin Nano (6-core Arm® Cortex®-A78AE v8.2 64-bit CPU)
* **GPU 架構**: NVIDIA Ampere 架構，具備 1024 個 CUDA 核心與 32 個 Tensor 核心
* **作業系統**: Ubuntu 22.04 LTS (JetPack 6.0)
* **軟體環境**: CUDA 12.6, TensorRT 10.x, PyTorch 2.x
* **測試模型**: Transformer-based Fast Style Transfer (Mosaic style)
* **輸入維度**: 480 x 640 (RGB)

## 3. 實驗方法 (Methodology)
### 3.1 模型轉換流程
1. **PyTorch to ONNX**: 將訓練好的 `.pth` 權重匯出為 ONNX 格式，固定輸入尺寸為 480x640。
2. **TensorRT 編譯**:
   * **FP16 量化**: 開啟 `trt.BuilderFlag.FP16`，針對硬體 Tensor Cores 進行算子融合。
   * **INT8 量化**: 採用 Post-Training Quantization (PTQ)，實作 `IInt8EntropyCalibrator2` 並使用 `content.png` 作為校準參考圖。
3. **記憶體管理**: 使用 `ctypes` 呼叫 `libcudart.so` 進行 GPU 記憶體分配與 CUDA Stream 控制。

### 3.2 效能測試指標
* **Throughput (FPS)**: 每秒處理幀數。
* **Latency (ms)**: 單張影像推論之平均延遲時間。

## 4. 實驗結果 (Results)

| 測試階段 | 軟硬體配置 | 量化精度 | 平均延遲 (ms) | 換算 FPS |
| :--- | :--- | :--- | :--- | :--- |
| **Baseline** | PC (Local Python) | FP32 | ~170.0 | 5.88 |
| **Stage 1** | Orin Nano (Default) | FP16 | 46.93 | **21.31** |
| **Stage 2** | Orin Nano (MAXN) | INT8 | 64.31 | 15.55 |
| **Stage 3** | Orin Nano (MAXN) | FP16 | 68.95 | 14.50 |

## 5. 結果分析與討論 (Discussion)

### 5.1 INT8 量化之效能逆增長分析
實驗發現 INT8 量化在 MAXN 模式下之 FPS (15.55) 顯著低於預設模式下之 FP16 (21.31)。經由 TensorRT 編譯日誌分析，其成因為：
1. **格式轉換開銷 (Reformatting Overhead)**: 模型中包含大量 `InstanceNorm2d` 層，此算子在 INT8 精度下不穩定，TensorRT 強制回退（Fallback）至 FP16 執行。這導致資料在層間傳遞時需頻繁進行 INT8 與 FP16 格式轉換。
2. **記憶體牆 (Memory Bound)**: 處理 480x640 解析度影像時，特徵圖數據量極大。Orin Nano 之記憶體頻寬（約 68 GB/s）成為主要瓶頸，單純提升運算核心算力（INT8）無法抵銷搬運數據的時間成本。

### 5.2 MAXN 模式下的降頻現象
在開啟 `nvpmodel -m 0` 與 `jetson_clocks` 後，FPS 從 21 掉落至 15。初步研判為：
* **功耗牆 (Power Throttling)**: 極限負載下，瞬間電流需求可能觸發 PMIC 保護機制，導致硬體強制降頻。
* **核心競爭**: 在高時脈下，CPU 與 GPU 對記憶體控制器的競爭加劇，延遲反而增加。

## 6. 結論 (Conclusion)
針對 480x640 高解析度之風格轉換模型，在 Jetson Orin Nano 上的效能最佳實作路徑為 **「預設功耗模式 + TensorRT FP16 量化 + 非同步推論」**。此配置下可穩定達到 21.31 FPS，最接近即時運算（30 FPS）需求。後續研究可朝向模型架構輕量化（如減少通道數）或分區推論方向進行，以突破記憶體頻寬限制。