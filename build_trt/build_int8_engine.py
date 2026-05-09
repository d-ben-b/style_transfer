import tensorrt as trt
import os
import numpy as np
from PIL import Image
import ctypes
import ctypes.util

# ==========================================
# 1. 底層 CUDA C API 記憶體控制 (解決環境問題)
# ==========================================
cudart_path = ctypes.util.find_library("cudart")
if not cudart_path:
    cudart_path = "/usr/local/cuda/lib64/libcudart.so"
cudart = ctypes.CDLL(cudart_path)

def cuda_malloc(size):
    ptr = ctypes.c_void_p()
    cudart.cudaMalloc(ctypes.byref(ptr), ctypes.c_size_t(size))
    return ptr.value

def cuda_free(ptr):
    cudart.cudaFree(ctypes.c_void_p(ptr))

def cuda_memcpy_htod(dst, src_array, size):
    # cudaMemcpyHostToDevice 的參數代碼是 1
    cudart.cudaMemcpy(ctypes.c_void_p(dst), src_array.ctypes.data_as(ctypes.c_void_p), ctypes.c_size_t(size), ctypes.c_int(1))

# ==========================================
# 2. INT8 資料校準器 (Calibrator) 核心實作
# ==========================================
class ImageCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, image_path, shape=(1, 3, 480, 640), cache_file="mosaic_calibration.cache"):
        trt.IInt8EntropyCalibrator2.__init__(self)
        self.cache_file = cache_file
        self.shape = shape
        self.batch_size = shape[0]
        
        # 準備校準用的圖片資料
        img = Image.open(image_path).convert("RGB").resize((shape[3], shape[2]))
        img_arr = np.array(img, dtype=np.float32)
        img_arr = np.transpose(img_arr, (2, 0, 1)) # HWC 轉 CHW
        img_arr = np.expand_dims(img_arr, axis=0)  # 增加 Batch 維度
        
        # 確保記憶體連續，並分配 GPU 記憶體
        self.batch_data = np.ascontiguousarray(img_arr, dtype=np.float32)
        self.device_input = cuda_malloc(self.batch_data.nbytes)
        
        self.current_index = 0
        self.max_batches = 10 # 模擬送入 10 次 Batch 進行統計校準

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names):
        if self.current_index >= self.max_batches:
            return None # 校準結束
        # 將 CPU 上的圖片資料複製到 GPU
        cuda_memcpy_htod(self.device_input, self.batch_data, self.batch_data.nbytes)
        self.current_index += 1
        return [int(self.device_input)] # 回傳 GPU 記憶體指標

    def read_calibration_cache(self):
        # 如果已經有算好的校準檔就直接讀取，省時間
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)

    def free(self):
        cuda_free(self.device_input)

# ==========================================
# 3. 編譯 INT8 TensorRT 引擎
# ==========================================
def build_int8_engine(onnx_path, engine_path, calib_image):
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)

    print(f"開始讀取 ONNX: {onnx_path}")
    with open(onnx_path, "rb") as f:
        parser.parse(f.read())

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 * (1 << 30)) # 2GB 暫存

    # ====== 啟動 INT8 與校準器 ======
    if builder.platform_has_fast_int8:
        config.set_flag(trt.BuilderFlag.INT8)
        calibrator = ImageCalibrator(calib_image)
        config.int8_calibrator = calibrator
        print("已啟用 INT8 精度優化與動態範圍校準！")
    else:
        print("警告：此硬體不支援 INT8！")
        return
    # ===============================

    print("正在編譯 INT8 引擎 (需包含校準時間，請等候約 3~8 分鐘)...")
    engine = builder.build_serialized_network(network, config)
    
    with open(engine_path, "wb") as f:
        f.write(engine)
        
    calibrator.free() # 釋放 GPU 記憶體
    print(f"引擎編譯成功！已存檔為: {engine_path}")

if __name__ == "__main__":
    # 使用你的 content.png 作為校準參考圖
    build_int8_engine("mosaic_480x640.onnx", "mosaic_int8.engine", "content.png")