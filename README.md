# Edge Computing Neural Style Transfer Demo

This project provides a real-time hardware-accelerated Neural Style Transfer (NST) application using TensorRT and Gradio. It is designed to run efficiently on edge computing devices like the NVIDIA Jetson Orin series.

## Features

The web interface (`app.py`) is divided into three main tabs to suit different demonstration and testing scenarios:

1. **🎥 Orin Local Camera Streaming**:
   Directly captures video from a USB camera (`/dev/video0`) plugged into the edge device (e.g., Jetson Orin). Best for physical setups with dedicated displays.
2. **🌐 Web-based Real-Time WebRTC Streaming**:
   Allows you to use the local camera of your smartphone or laptop over the network to send frames to the Orin Nano for inference.
3. **🖼️ Static Image Analysis**:
   Upload local images, paste from the clipboard, or take a single snapshot to measure exact inference latency (ms) and FPS.

## Prerequisites

Ensure you have your virtual environment activated and the required dependencies installed:

```bash
# Activate your virtual environment (Windows PowerShell example)
.venv\Scripts\Activate.ps1

# Install requirements if not already done
pip install -r requirements.txt
```

### Required TensorRT Engines

The application expects the following converted TensorRT engines to be present in the same directory as `app.py`:

- `mosaic_fp16.engine`
- `mosaic_int8.engine`

_(Note: The app will gracefully continue if one engine fails to load, but at least one must be available for inference.)_

## Usage

Start the server by executing:

```bash
python app.py
```

Once the server has started, it will bind to `0.0.0.0` on port `7860`. You can access the user interface from your web browser:

- **Local Access:** `http://localhost:7860` or `http://127.0.0.0:7860`
- **Network Access:** `http://<YOUR_DEVICE_IP_ADDRESS>:7860`

### Options

Inside the web app, you can seamlessly switch the **Inference Backend** between `TensorRT (FP16)` and `TensorRT (INT8)` using the radio buttons to compare latency and output quality on the fly.
