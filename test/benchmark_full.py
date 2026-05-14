import os
import csv
import time
import numpy as np
import tensorrt as trt
import ctypes
import ctypes.util
import subprocess

# ==========================================
# 1. 系統資訊與硬體偵測模組
# ==========================================
def get_gpu_name():
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], 
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        name = result.stdout.strip()
        return name if name else "Unknown GPU"
    except Exception:
        return "Unknown GPU (nvidia-smi failed)"

def get_power_mode():
    try:
        result = subprocess.run(
            ['nvpmodel', '-q'], 
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        for line in result.stdout.split('\n'):
            if "NV Power Mode" in line or "Current Mode" in line:
                return line.split(':')[-1].strip()
        return "Unknown Mode"
    except Exception:
        return "Unknown Mode (nvpmodel failed)"

def get_trt_version():
    return trt.__version__

# ==========================================
# 2. CUDA 底層 API 封裝 (供 TensorRT 使用)
# ==========================================
cudart_path = ctypes.util.find_library("cudart") or "/usr/local/cuda/lib64/libcudart.so"
cudart = ctypes.CDLL(cudart_path)

def cuda_malloc(size):
    ptr = ctypes.c_void_p()
    if cudart.cudaMalloc(ctypes.byref(ptr), ctypes.c_size_t(size)) != 0:
        raise RuntimeError("cudaMalloc failed!")
    return ptr.value

def cuda_free(ptr):
    cudart.cudaFree(ctypes.c_void_p(ptr))

def cuda_stream_create():
    stream = ctypes.c_void_p()
    cudart.cudaStreamCreate(ctypes.byref(stream))
    return stream.value

def cuda_stream_synchronize(stream):
    cudart.cudaStreamSynchronize(ctypes.c_void_p(stream))

def cuda_stream_destroy(stream):
    cudart.cudaStreamDestroy(ctypes.c_void_p(stream))

def cuda_memcpy_htod_async(dst_ptr, src_array, size, stream):
    cudart.cudaMemcpyAsync(ctypes.c_void_p(dst_ptr), src_array.ctypes.data_as(ctypes.c_void_p),
                           ctypes.c_size_t(size), ctypes.c_int(1), ctypes.c_void_p(stream))

def cuda_memcpy_dtoh_async(dst_array, src_ptr, size, stream):
    cudart.cudaMemcpyAsync(dst_array.ctypes.data_as(ctypes.c_void_p), ctypes.c_void_p(src_ptr),
                           ctypes.c_size_t(size), ctypes.c_int(2), ctypes.c_void_p(stream))

class TRTEngineWrapper:
    def __init__(self, engine_path):
        self.logger = trt.Logger(trt.Logger.ERROR)
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        
        self.context = self.engine.create_execution_context()
        self.stream = cuda_stream_create()
        
        self.shape = (1, 3, 480, 640)
        self.buffer_size = 1 * 3 * 480 * 640 * 4 
        
        self.d_input = cuda_malloc(self.buffer_size)
        self.d_output = cuda_malloc(self.buffer_size)
        
        in_name = self.engine.get_tensor_name(0)
        out_name = self.engine.get_tensor_name(1)
        self.context.set_tensor_address(in_name, self.d_input)
        self.context.set_tensor_address(out_name, self.d_output)
        
        self.h_output = np.empty(self.shape, dtype=np.float32)

    def infer(self, h_input):
        start_time = time.time()
        cuda_memcpy_htod_async(self.d_input, h_input, self.buffer_size, self.stream)
        self.context.execute_async_v3(stream_handle=self.stream)
        cuda_memcpy_dtoh_async(self.h_output, self.d_output, self.buffer_size, self.stream)
        cuda_stream_synchronize(self.stream)
        return time.time() - start_time

    def __del__(self):
        cuda_free(self.d_input)
        cuda_free(self.d_output)
        cuda_stream_destroy(self.stream)

# ==========================================
# 3. 量化演算法模擬模組 (完全改寫為純 Numpy)
# ==========================================
def create_mock_weights(shape=(1000, 1000)):
    # 使用 Numpy 產生常態分佈
    return np.random.randn(*shape).astype(np.float32)

def linear_quantize(weights, bit_width=8):
    abs_max = np.abs(weights).max()
    q_max = (1 << (bit_width - 1)) - 1 
    scale = abs_max / q_max
    # Numpy 的 clip 和 round
    quantized_weights = np.clip(np.round(weights / scale), -q_max, q_max)
    return quantized_weights * scale

def power_of_two_quantize(weights, bit_width=8):
    epsilon = 1e-7
    abs_weights = np.abs(weights) + epsilon
    # Numpy 的 log2 和 round
    exponents = np.round(np.log2(abs_weights))
    
    min_exp = -( (1 << (bit_width - 1)) - 1 )
    max_exp = 0 
    exponents = np.clip(exponents, min_exp, max_exp)
    
    po2_weights = (2.0 ** exponents) * np.sign(weights)
    po2_weights[abs_weights < (2.0 ** min_exp)] = 0.0
    return po2_weights

def compute_mse(original, quantized):
    return ((original - quantized) ** 2).mean()

# ==========================================
# 4. 主執行流程
# ==========================================
def main():
    log_dir = "./log"
    os.makedirs(log_dir, exist_ok=True)
    hw_csv = os.path.join(log_dir, "hardware_benchmark.csv")
    quant_csv = os.path.join(log_dir, "quantization_simulation.csv")
    accel_csv = os.path.join(log_dir, "acceleration_simulation.csv")

    print("=" * 60)
    print(" 🚀 Edge AI 專題綜合測試腳本 (Hardware + Theory + Acceleration)")
    print("=" * 60)

    # ---------------------------------------------------------
    # PART A: 實體硬體推論效能測試
    # ---------------------------------------------------------
    print("\n[PART A: 實體硬體 TensorRT 推論測試]")
    gpu_name = get_gpu_name()
    power_mode = get_power_mode()
    trt_version = get_trt_version()
    print(f"硬體: {gpu_name} | 模式: {power_mode} | TRT: {trt_version}")
    
    dummy_input = np.ascontiguousarray(np.random.randn(1, 3, 480, 640).astype(np.float32))
    
    engines = [
        {"path": "mosaic_fp16.engine", "quantization": "FP16"},
        {"path": "mosaic_fp16_new.engine", "quantization": "FP16 (Recompiled)"},
        {"path": "mosaic_int8.engine", "quantization": "INT8"}
    ]
    
    hw_results = []
    for item in engines:
        if not os.path.exists(item["path"]): continue
        print(f"測試 {item['quantization']} 引擎...")
        try:
            wrapper = TRTEngineWrapper(item["path"])
            for _ in range(50): wrapper.infer(dummy_input) # Warm-up
            
            latencies = [wrapper.infer(dummy_input) * 1000 for _ in range(200)]
            avg_lat, std_lat = np.mean(latencies), np.std(latencies)
            fps = 1000.0 / avg_lat
            
            hw_results.append({
                "GPU_Name": gpu_name, "Power_Mode": power_mode, 
                "Model_File": item["path"], "Quantization": item["quantization"],
                "Avg_Latency_ms": f"{avg_lat:.2f}", "Std_Latency_ms": f"{std_lat:.2f}", 
                "FPS": f"{fps:.2f}"
            })
            print(f"  -> FPS: {fps:.2f} | Avg: {avg_lat:.2f} ms")
            del wrapper
        except Exception as e: print(f"測試失敗: {e}")

    if hw_results:
        with open(hw_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=hw_results[0].keys())
            writer.writeheader()
            writer.writerows(hw_results)

    # ---------------------------------------------------------
    # PART B: 任意位元 (Arbitrary Bit-width) 量化演算法理論模擬
    # ---------------------------------------------------------
    print("\n[PART B: 任意位元量化演算法理論模擬 (包含 INT8, INT6, INT4, INT2)]")
    fp32_weights = create_mock_weights((1000, 1000))
    
    quant_results = []
    
    # 測試不同的位元寬度
    bit_widths_to_test = [8, 6, 4, 2]
    
    for bw in bit_widths_to_test:
        # 計算 Linear 量化
        linear_mse = compute_mse(fp32_weights, linear_quantize(fp32_weights, bit_width=bw))
        quant_results.append({
            "Bit_Width": f"{bw}-bit",
            "Quantization_Type": "Linear",
            "MSE_Loss": f"{linear_mse:.6f}",
            "Description": f"均勻切割成 {2**bw} 階"
        })
        
        # 計算 Power-of-2 量化
        po2_mse = compute_mse(fp32_weights, power_of_two_quantize(fp32_weights, bit_width=bw))
        quant_results.append({
            "Bit_Width": f"{bw}-bit",
            "Quantization_Type": "Power-of-2",
            "MSE_Loss": f"{po2_mse:.6f}",
            "Description": f"對數分佈，限縮至 {bw}-bit 指數"
        })

    # 寫入 CSV
    with open(quant_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["Bit_Width", "Quantization_Type", "MSE_Loss", "Description"])
        writer.writeheader()
        writer.writerows(quant_results)
        
    print("已完成量化誤差模擬測試！")
    for r in quant_results:
        print(f"  -> {r['Bit_Width']} {r['Quantization_Type']:>10}: MSE = {r['MSE_Loss']}")

    # ---------------------------------------------------------
    # PART C: 底層運算元加速模擬 (Speedup Simulation)
    # ---------------------------------------------------------
    print("\n[PART C: 運算元層級效能模擬 (乘法器 vs 位移器)]")
    N = 50_000_000 # 模擬 5000 萬次運算
    
    activations = np.random.randint(-128, 127, size=N, dtype=np.int32)
    weights_linear = np.random.randint(-128, 127, size=N, dtype=np.int32)
    weights_po2_shift = np.random.randint(0, 7, size=N, dtype=np.int32)
    
    start_time = time.time()
    np.multiply(activations, weights_linear)
    time_mul = time.time() - start_time
    
    start_time = time.time()
    np.left_shift(activations, weights_po2_shift)
    time_shift = time.time() - start_time
    
    speedup = time_mul / time_shift
    print(f"[Linear INT8] 執行 5000 萬次整數乘法耗時: {time_mul:.4f} 秒")
    print(f"[Power-of-2]  執行 5000 萬次位移運算耗時: {time_shift:.4f} 秒")
    print(f"🎯 預估硬體架構轉換加速比 (Speedup): {speedup:.2f}x")

    accel_results = [{
        "Operation": "Multiply (Linear INT8)", "Time_Seconds": f"{time_mul:.4f}", "Speedup": "1.00x"
    }, {
        "Operation": "Bit-shift (Power-of-2)", "Time_Seconds": f"{time_shift:.4f}", "Speedup": f"{speedup:.2f}x"
    }]
    
    with open(accel_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["Operation", "Time_Seconds", "Speedup"])
        writer.writeheader()
        writer.writerows(accel_results)

    print("\n" + "=" * 60)
    print(f"✅ 測試完成！請至 ./log 目錄查看 3 份 CSV 報表：")
    print("   1. hardware_benchmark.csv (實體 FPS 數據)")
    print("   2. quantization_simulation.csv (量化理論 MSE 誤差)")
    print("   3. acceleration_simulation.csv (硬體架構改進加速比)")

if __name__ == "__main__":
    main()