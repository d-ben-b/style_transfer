import tensorrt as trt
import os

# 1. 初始化 TensorRT 的日誌記錄器
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def build_engine(onnx_file_path, engine_file_path):
    print(f"開始讀取 ONNX 檔案: {onnx_file_path}")
    
    # 2. 建立 Builder 與 Network
    builder = trt.Builder(TRT_LOGGER)
    # EXPLICIT_BATCH 是目前 TensorRT 處理 ONNX 的標準設定
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    
    # 3. 建立 ONNX 解析器並載入模型
    parser = trt.OnnxParser(network, TRT_LOGGER)
    if not os.path.exists(onnx_file_path):
        print("找不到 ONNX 檔案，請確認路徑！")
        return
        
    with open(onnx_file_path, "rb") as model:
        if not parser.parse(model.read()):
            print("ONNX 解析失敗:")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return
            
    print("ONNX 解析成功！開始配置硬體優化參數...")
    
    # 4. 配置編譯參數 (BuilderConfig)
    config = builder.create_builder_config()
    # 設定編譯時可使用的最大工作記憶體 (這裡設為 2GB)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 * (1 << 30))
    
    # ====== 核心優化設定 ======
    # 檢查硬體是否支援 FP16 並開啟
    if builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("已開啟 FP16 精度優化 (Hardware supports fast FP16)")
    else:
        print("警告: 此硬體不支援快速 FP16，將以 FP32 編譯")
    # ==========================
    
    # 5. 開始編譯引擎 (這步驟會花費數分鐘，因為會進行 Kernel Auto-Tuning)
    print("正在編譯 TensorRT 引擎，請耐心等候 (約需 2~5 分鐘)...")
    serialized_engine = builder.build_serialized_network(network, config)
    
    if serialized_engine is None:
        print("引擎編譯失敗！")
        return
        
    # 6. 將編譯好的引擎存檔
    with open(engine_file_path, "wb") as f:
        f.write(serialized_engine)
    print(f"引擎編譯成功！已存檔為: {engine_file_path}")

if __name__ == "__main__":
    onnx_path = "mosaic_480x640.onnx"
    engine_path = "mosaic_fp16.engine"
    build_engine(onnx_path, engine_path)