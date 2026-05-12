import cv2
import numpy as np
import tensorrt as trt
import ctypes
import ctypes.util
import time

# ==========================================
# 1. 底層 CUDA API 呼叫
# ==========================================
cudart_path = ctypes.util.find_library("cudart")
if not cudart_path:
    cudart_path = "/usr/local/cuda/lib64/libcudart.so"
cudart = ctypes.CDLL(cudart_path)


def cuda_malloc(size):
    ptr = ctypes.c_void_p()
    res = cudart.cudaMalloc(ctypes.byref(ptr), ctypes.c_size_t(size))
    if res != 0:
        raise RuntimeError(f"cudaMalloc failed! {res}")
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
# 2. 攝影機推論主程式 (寫入影片版)
# ==========================================
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


def run_live_style_transfer(
    engine_path, output_filename="output_style.mp4", record_seconds=10
):
    print(f"正在載入 TensorRT 引擎: {engine_path}")
    with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())

    context = engine.create_execution_context()

    BATCH_SIZE, CHANNELS, HEIGHT, WIDTH = 1, 3, 480, 640
    buffer_size = BATCH_SIZE * CHANNELS * HEIGHT * WIDTH * 4

    d_input = cuda_malloc(buffer_size)
    d_output = cuda_malloc(buffer_size)
    stream = cuda_stream_create()

    in_name, out_name = engine.get_tensor_name(0), engine.get_tensor_name(1)
    context.set_tensor_address(in_name, d_input)
    context.set_tensor_address(out_name, d_output)

    h_output = np.empty((BATCH_SIZE, CHANNELS, HEIGHT, WIDTH), dtype=np.float32)

    # 初始化攝影機
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("錯誤：無法開啟攝影機。")
        return

    # 設定影片寫入器 (VideoWriter)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # 使用 mp4v 編碼
    # 假設我們預期的 FPS 大約是 20
    out = cv2.VideoWriter(output_filename, fourcc, 20.0, (WIDTH, HEIGHT))

    print(f"開始錄製 {record_seconds} 秒的風格轉換影片...")

    start_time = time.time()
    frame_count = 0

    while (time.time() - start_time) < record_seconds:
        ret, frame = cap.read()
        if not ret:
            break

        frame_resized = cv2.resize(frame, (WIDTH, HEIGHT))
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        frame_chw = np.transpose(frame_rgb, (2, 0, 1))
        h_input = np.ascontiguousarray(
            np.expand_dims(frame_chw, axis=0).astype(np.float32)
        )

        # TensorRT 推論
        cuda_memcpy_htod_async(d_input, h_input, buffer_size, stream)
        context.execute_async_v3(stream_handle=stream)
        cuda_memcpy_dtoh_async(h_output, d_output, buffer_size, stream)
        cuda_stream_synchronize(stream)

        # 後處理
        out_img = h_output[0]
        out_img = np.clip(out_img, 0, 255).astype(np.uint8)
        out_img = np.transpose(out_img, (1, 2, 0))
        out_img_bgr = cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)

        # 寫入影片檔
        out.write(out_img_bgr)
        frame_count += 1

    actual_fps = frame_count / (time.time() - start_time)
    print(f"錄製完成！總共寫入 {frame_count} 幀，平均處理速度: {actual_fps:.2f} FPS")
    print(f"影片已儲存為: {output_filename}")

    cap.release()
    out.release()
    cuda_free(d_input)
    cuda_free(d_output)
    cuda_stream_destroy(stream)


if __name__ == "__main__":
    run_live_style_transfer("mosaic_fp16.engine")
