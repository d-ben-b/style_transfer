import gradio as gr
import cv2
import numpy as np
import tensorrt as trt
import ctypes
import ctypes.util
import time

# ==========================================
# 1. CUDA 底層 API 封裝
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

# ==========================================
# 2. TensorRT 引擎封裝類別
# ==========================================
class TRTEngineWrapper:
    def __init__(self, engine_path):
        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        
        self.context = self.engine.create_execution_context()
        self.stream = cuda_stream_create()
        
        # 固定尺寸
        self.shape = (1, 3, 480, 640)
        self.buffer_size = 1 * 3 * 480 * 640 * 4 
        
        self.d_input = cuda_malloc(self.buffer_size)
        self.d_output = cuda_malloc(self.buffer_size)
        
        in_name = self.engine.get_tensor_name(0)
        out_name = self.engine.get_tensor_name(1)
        self.context.set_tensor_address(in_name, self.d_input)
        self.context.set_tensor_address(out_name, self.d_output)
        
        self.h_output = np.empty(self.shape, dtype=np.float32)

    def infer(self, img_chw):
        h_input = np.ascontiguousarray(img_chw)
        
        start_time = time.time()
        cuda_memcpy_htod_async(self.d_input, h_input, self.buffer_size, self.stream)
        self.context.execute_async_v3(stream_handle=self.stream)
        cuda_memcpy_dtoh_async(self.h_output, self.d_output, self.buffer_size, self.stream)
        cuda_stream_synchronize(self.stream)
        infer_time = time.time() - start_time
        
        return self.h_output[0], infer_time

    def __del__(self):
        cuda_free(self.d_input)
        cuda_free(self.d_output)
        cuda_stream_destroy(self.stream)

# ==========================================
# 3. 系統初始化與推論邏輯
# ==========================================
print("正在載入 TensorRT 引擎...")

trt_fp16_engine = None
trt_int8_engine = None

try:
    trt_fp16_engine = TRTEngineWrapper("mosaic_fp16.engine")
    print("✅ FP16 引擎載入成功")
except Exception as e:
    print(f"⚠️ FP16 引擎載入失敗: {e}")

try:
    trt_int8_engine = TRTEngineWrapper("mosaic_int8.engine")
    print("✅ INT8 引擎載入成功")
except Exception as e:
    print(f"⚠️ INT8 引擎載入失敗: {e}")

def process_image(input_img, backend_choice):
    if input_img is None:
        return None, "請上傳圖片"

    img_resized = cv2.resize(input_img, (640, 480))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_chw = np.transpose(img_rgb, (2, 0, 1)).astype(np.float32)
    img_batch = np.expand_dims(img_chw, axis=0)

    try:
        if backend_choice == "TensorRT (FP16)":
            if not trt_fp16_engine: return None, "找不到 FP16 引擎"
            out_img_chw, infer_time = trt_fp16_engine.infer(img_batch)

        elif backend_choice == "TensorRT (INT8)":
            if not trt_int8_engine: return None, "找不到 INT8 引擎"
            out_img_chw, infer_time = trt_int8_engine.infer(img_batch)
            
        else:
            return None, "未知的後端選擇"

        out_img_hwc = np.transpose(out_img_chw, (1, 2, 0))
        out_img_clamped = np.clip(out_img_hwc, 0, 255).astype(np.uint8)
        
        fps = 1.0 / infer_time
        stats = f"**推論耗時**: {infer_time*1000:.2f} ms\n**換算 FPS**: {fps:.2f} 幀/秒"
        
        return out_img_clamped, stats

    except Exception as e:
        return None, f"發生錯誤: {str(e)}"

# ==========================================
# 4. Gradio UI 介面設計
# ==========================================
with gr.Blocks(title="Edge Computing NST Demo", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🚀 邊緣運算裝置影像風格轉換效能展示")
    
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(label="上傳測試影像", type="numpy")
            backend_radio = gr.Radio(
                choices=["TensorRT (FP16)", "TensorRT (INT8)"],
                value="TensorRT (FP16)",
                label="Inference Backend"
            )
            submit_btn = gr.Button("執行風格轉換", variant="primary")
            
        with gr.Column():
            output_image = gr.Image(label="轉換結果 (480x640)")
            output_stats = gr.Markdown(label="效能指標")

    submit_btn.click(
        fn=process_image,
        inputs=[input_image, backend_radio],
        outputs=[output_image, output_stats]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)