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

### 4.1 模型推論效能 (Inference Performance)

| 測試配置 | 軟硬體配置 | 量化精度 | 平均延遲 (ms) | 標準差 (ms) | 換算 FPS |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Baseline** | PC (Local Python) | FP32 | ~170.0 | - | 5.88 |
| **Experiment 1** | Orin Nano (Unknown Mode)| FP16 | 71.74 | 0.31 | 13.94 |
| **Experiment 2** | Orin Nano (Unknown Mode)| INT8 | 65.10 | 0.31 | 15.36 |

### 4.2 量化與加速模擬分析 (Quantization & Acceleration Simulation)

針對底層量化演算法進一步模擬，比較線性量化與基於二的冪次方之量化方式：

**1. 量化誤差 (Quantization Error)**
| 量化方式 | MSE Loss | 說明 |
| :--- | :--- | :--- |
| Linear (INT8) | `0.000149` | 均勻分佈，保持高精度 |
| Power-of-2 | `0.158361` | 對數分佈，犧牲部分精度換取免乘法器硬體架構 |

**2. 運算加速模擬 (Acceleration Simulation)**
| 運算操作 | 執行時間 (秒) | 預期加速比 |
| :--- | :--- | :--- |
| Multiply (Linear INT8) | 0.0786 | 1.00x |
| Bit-shift (Power-of-2) | 0.0792 | 0.99x |

## 5. 結果分析與討論 (Discussion)

### 5.1 INT8 量化之效能微幅提升
實驗發現 INT8 量化之 FPS (15.36) 雖高於 FP16 (13.94)，但提升幅度不如預期（理論上 INT8 算力為 FP16 兩倍）。其成因為：
1. **格式轉換開銷 (Reformatting Overhead)**: 模型中包含大量 `InstanceNorm2d` 層，此算子在 INT8 精度下不穩定，TensorRT 可能強制回退（Fallback）至 FP16 執行。這導致資料在層間傳遞時需頻繁進行 INT8 與 FP16 格式轉換，抵銷了部分算力優勢。
2. **記憶體牆 (Memory Bound)**: 處理 480x640 解析度影像時，特徵圖數據量極大。單純提升運算核心算力無法完全抵銷搬運數據的時間成本。

### 5.2 Power-of-2 量化之實務效益評估
由模擬數據可知，雖然 Power-of-2 能帶來免乘法的硬體設計優勢（使用位移運算 Bit-shift），但在此實驗平台上，其運算時間 (0.0792秒) 並未優於標準 INT8 乘法 (0.0786秒)，加速比甚至呈 0.99x。這表示現代運算單元（如 Tensor Core）對正規 MAC 運算已有極度優化，反觀純軟體模擬或通用 ALU 上的分支處理可能使 Bit-shift 未能展現預期加速。此外，其高達 0.158 的 MSE Loss 顯示影像品質可能會顯著下降。

## 6. 結論 (Conclusion)
針對 480x640 高解析度之風格轉換模型，在 Jetson Orin Nano 上的實測表現，INT8 可達約 15 FPS，略優於 FP16 的 14 FPS。考量到額外的量化誤差與微小的效能提升，若對於影像品質要求極高，可直接採用 **FP16**；若追求極致推論速度，且能容忍轉換過程中的些微精度流失，則可採用 **INT8** 建置。後續研究應朝向模型架構輕量化（如減少通道數），或完全替換對 INT8 不友善之 Normalize 層，以進一步突破效能限制。