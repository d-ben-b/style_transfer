import tensorrt as trt
import ctypes
import ctypes.util
import time

# ==========================================
# 1. 呼叫系統底層 CUDA C API (包含 Stream)
# ==========================================
cudart_path = ctypes.util.find_library("cudart")
if not cudart_path:
    cudart_path = "/usr/local/cuda/lib64/libcudart.so"
cudart = ctypes.CDLL(cudart_path)

def cuda_malloc(size):
    ptr = ctypes.c_void_p()
    res = cudart.cudaMalloc(ctypes.byref(ptr), ctypes.c_size_t(size))
    if res != 0: raise RuntimeError(f"cudaMalloc failed! {res}")
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

# ==========================================
# 2. TensorRT 10.x 非同步推論測試
# ==========================================
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def load_engine(engine_file_path):
    with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

def test_inference_speed(engine_path):
    print(f"正在載入 TensorRT 引擎: {engine_path}")
    engine = load_engine(engine_path)
    context = engine.create_execution_context()

    buffer_size = 1 * 3 * 480 * 640 * 4

    print("正在配置 GPU 記憶體與資料流 (CUDA Stream)...")
    d_input = cuda_malloc(buffer_size)
    d_output = cuda_malloc(buffer_size)
    stream = cuda_stream_create()

    # ====== TensorRT 10.x 新版 API 記憶體綁定 ======
    # 動態獲取 ONNX 模型中的輸入/輸出節點名稱
    input_name = engine.get_tensor_name(0)
    output_name = engine.get_tensor_name(1)

    # 針對名稱進行記憶體綁定
    context.set_tensor_address(input_name, d_input)
    context.set_tensor_address(output_name, d_output)
    # ===============================================

    print("開始進行暖機 (Warm-up)...")
    for _ in range(10):
        # 新版 API 只吃 stream_handle，不再吃 bindings
        context.execute_async_v3(stream_handle=stream)
    cuda_stream_synchronize(stream)

    print("開始測量推論速度...")
    num_iterations = 100
    start_time = time.time()

    for _ in range(num_iterations):
        context.execute_async_v3(stream_handle=stream)
    cuda_stream_synchronize(stream)

    end_time = time.time()
    total_time = end_time - start_time
    avg_time = total_time / num_iterations
    fps = 1.0 / avg_time

    print(f"測試完成！總共執行 {num_iterations} 次")
    print(f"平均單張影像推論時間: {avg_time * 1000:.2f} 毫秒 (ms)")
    print(f"換算 FPS: {fps:.2f} 幀/秒")

    cuda_free(d_input)
    cuda_free(d_output)
    cuda_stream_destroy(stream)

if __name__ == "__main__":
    # 先測試 INT8 引擎的極限速度
    test_inference_speed("mosaic_fp16.engine")