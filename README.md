# SmartGuard AI

SmartGuard AI is a real-time entrance security system built with Flask, OpenCV, and YOLOv8n that uses a smartphone IP camera and multi-angle facial recognition to identify family members, detect packages, and send instant Telegram alerts when an unknown visitor loiters for over 15 seconds.

## Features

- **Live Smartphone IP Camera Streaming:** Multi-threaded MJPEG pipeline separating frame capture, YOLOv8n + ByteTrack inference, and smooth video rendering.
- **360° Face ID Enrollment & Recognition:** Interactive multi-angle facial registration using OpenCV YuNet and SFace with L2-normalized centroid matching and multi-frame voting.
- **Smart Loitering & Package Alerts:** Triggers instant Telegram snapshot notifications only when an unrecognized visitor stays longer than 15 seconds or a package arrives.
- **Low-Light Enhancement:** Real-time Gaussian denoising, Gamma correction, and CLAHE contrast enhancement for dark entrances.

## Setup & Run

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
