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

# 控制 Orin 本機串流的開關
streaming_active = False

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
    cudart.cudaMemcpyAsync(
        ctypes.c_void_p(dst_ptr),
        src_array.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_size_t(size),
        ctypes.c_int(1),
        ctypes.c_void_p(stream),
    )

def cuda_memcpy_dtoh_async(dst_array, src_ptr, size, stream):
    cudart.cudaMemcpyAsync(
        dst_array.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_void_p(src_ptr),
        ctypes.c_size_t(size),
        ctypes.c_int(2),
        ctypes.c_void_p(stream),
    )

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
        cuda_memcpy_dtoh_async(
            self.h_output, self.d_output, self.buffer_size, self.stream
        )
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
        return None, "請提供影像來源"

    # 統一影像尺寸與格式
    img_resized = cv2.resize(input_img, (640, 480))
    # 確保色彩通道順序正確 (Webcam 或 cv2 讀取的格式可能不同，此處統一轉為 RGB 推論)
    if len(img_resized.shape) == 3 and img_resized.shape[2] == 3:
        pass # 假設已經是正確格式
        
    img_chw = np.transpose(img_resized, (2, 0, 1)).astype(np.float32)
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
        
        fps = 1.0 / (infer_time + 1e-5) # 避免除以零
        stats = f"**推論耗時**: {infer_time*1000:.2f} ms\n**換算 FPS**: {fps:.2f} 幀/秒"

        return out_img_clamped, stats

    except Exception as e:
        return None, f"發生錯誤: {str(e)}"

# ==========================================
# 4. Orin 本機攝影機生成器邏輯
# ==========================================
def stream_from_orin_camera(backend_choice):
    global streaming_active
    streaming_active = True
    
    # 開啟 Orin 本機上的攝影機 (0 通常代表 /dev/video0)
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        yield None, "無法開啟 Orin 上的攝影機"
        return

    while streaming_active:
        ret, frame = cap.read()
        if not ret:
            break
            
        # OpenCV 讀取為 BGR，轉為 RGB 給模型
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 推論
        out_img, stats = process_image(frame_rgb, backend_choice)
        
        # 產出結果給網頁前端
        yield out_img, stats
        time.sleep(0.01)

    cap.release()

def stop_streaming():
    global streaming_active
    streaming_active = False
    return "串流已停止"

# ==========================================
# 5. Gradio UI 介面設計
# ==========================================
with gr.Blocks(title="Edge Computing NST Demo") as demo:
    gr.Markdown("# 🚀 邊緣運算裝置影像風格轉換效能展示")
    
    # 共用的後端選擇元件
    backend_radio = gr.Radio(
        choices=["TensorRT (FP16)", "TensorRT (INT8)"],
        value="TensorRT (INT8)",
        label="⚙️ 硬體推論後端選擇 (Inference Backend)",
    )

    with gr.Tabs():
        
        # ----------------------------------------
        # 分頁 1：Orin 本機視訊串流
        # ----------------------------------------
        with gr.TabItem("🎥 Orin 本機攝影機串流"):
            gr.Markdown("讀取插在 Jetson Orin 上的攝影機 (`/dev/video0`) 並進行即時推論。適合使用獨立螢幕展示。")
            with gr.Row():
                with gr.Column():
                    start_btn = gr.Button("▶️ 開啟 Orin 攝影機", variant="primary")
                    stop_btn = gr.Button("⏹️ 停止", variant="stop")
                with gr.Column():
                    local_live_output = gr.Image(label="即時風格轉換結果")
                    local_live_stats = gr.Markdown(label="效能指標")
            
            # 綁定按鈕事件 (使用 Generator 進行流式輸出)
            start_btn.click(
                fn=stream_from_orin_camera,
                inputs=[backend_radio],
                outputs=[local_live_output, local_live_stats]
            )
            stop_btn.click(
                fn=stop_streaming,
                outputs=[local_live_stats]
            )

        # ----------------------------------------
        # 分頁 2：網頁端即時串流 (WebRTC)
        # ----------------------------------------
        with gr.TabItem("🌐 網頁端即時串流 (Webcam)"):
            gr.Markdown("開啟您的筆電/手機攝影機，將畫面傳送至 Orin Nano 進行推論。適合遠端連線展示。")
            with gr.Row():
                with gr.Column():
                    # 開啟 streaming=True
                    web_live_input = gr.Image(sources=["webcam"], streaming=True, label="網頁攝影機畫面")
                with gr.Column():
                    web_live_output = gr.Image(label="即時風格轉換結果")
                    web_live_stats = gr.Markdown(label="效能指標")
            
            # 綁定即時事件
            web_live_input.stream(
                fn=process_image,
                inputs=[web_live_input, backend_radio],
                outputs=[web_live_output, web_live_stats],
            )

        # ----------------------------------------
        # 分頁 3：靜態圖片與單張拍照
        # ----------------------------------------
        with gr.TabItem("🖼️ 靜態圖片分析"):
            gr.Markdown("上傳本地圖片或拍攝單張照片，以進行最精確的硬體延遲 (Latency) 數據測量。")
            with gr.Row():
                with gr.Column():
                    static_input = gr.Image(sources=["upload", "webcam", "clipboard"], label="上傳測試影像")
                    submit_btn = gr.Button("執行風格轉換", variant="primary")
                with gr.Column():
                    static_output = gr.Image(label="轉換結果 (480x640)")
                    static_stats = gr.Markdown(label="效能指標")

            submit_btn.click(
                fn=process_image,
                inputs=[static_input, backend_radio],
                outputs=[static_output, static_stats],
            )

if __name__ == "__main__":
    # 將 theme 設定移至 launch() 內以符合 Gradio 6.0 規範
    demo.launch(server_name="0.0.0.0", server_port=7860, theme=gr.themes.Soft())