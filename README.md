## StereoCapture: Capturing Precise 3D Hand Landmarks with Stereo Vision

We present StereoCapture, a novel system designed to capture precise 3D hand landmarks using stereo vision. Leveraging state-of-the-art image-based and sensor-based tracking technologies, coupled with a cutting-edge hand gesture detection model, our system aims to accurately track the user's hand movements and gestures in real-time, enabling the collection of high-quality data for manipulation tasks. Our system demonstrates exceptional accuracy and robustness to noise, outperforming baseline methods in terms of accuracy and efficiency. Furthermore, our system's ability to track complex hand gestures and spatial relationships between the 3D hand landmarks underscores its versatility and applicability in various real-world scenarios. By demonstrating the efficacy of our system in capturing precise 3D hand landmarks, we aim to democratize access to robust datasets essential for advancing AI-driven robotics research and development.


### Usage
To use the StereoCapture system, follow these steps:

1. Make sure you have the necessary hardware components, including a stereo camera setup and a computer with sufficient processing power.
2. Install the required software dependencies, including OpenCV, MediaPipe, and other libraries.
3. Run `camera_test.py` to test the stereo camera setup; press `s` to save a stereo image pair.
4. When you have enough stereo image pairs, run `camera_calibration.py` to calibrate the stereo camera setup.
5. Run `run.py` to start the StereoCapture system and capture precise 3D hand landmarks in real-time.