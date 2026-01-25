"""Webcam capture module for image-based queries."""
import cv2
import base64
import numpy as np
from typing import Optional
import time


class WebcamCapture:
    """
    Handles webcam operations: open, capture, close.

    Designed to be opened briefly for capture, then immediately closed
    to avoid keeping the camera active unnecessarily.
    """

    def __init__(self, camera_index: int = 0, warmup_frames: int = 5):
        """
        Initialize webcam capture.

        Args:
            camera_index: Camera device index (0 = default camera)
            warmup_frames: Number of frames to skip for camera warmup
        """
        self.camera_index = camera_index
        self.warmup_frames = warmup_frames
        self._cap: Optional[cv2.VideoCapture] = None

    def capture(self) -> Optional[np.ndarray]:
        """
        Open camera, capture single frame, close camera.

        Returns:
            BGR image as numpy array, or None if capture failed
        """
        try:
            # Open camera
            self._cap = cv2.VideoCapture(self.camera_index)
            if not self._cap.isOpened():
                print("Error: Could not open webcam")
                return None

            # Warmup: skip first few frames (camera auto-exposure adjustment)
            for _ in range(self.warmup_frames):
                self._cap.read()

            # Capture frame
            ret, frame = self._cap.read()

            # Close camera immediately
            self._cap.release()
            self._cap = None

            if not ret or frame is None:
                print("Error: Could not capture frame")
                return None

            return frame

        except Exception as e:
            print(f"Error capturing from webcam: {e}")
            if self._cap:
                self._cap.release()
                self._cap = None
            return None

    def capture_as_base64(self, quality: int = 85) -> Optional[str]:
        """
        Capture image and return as base64-encoded JPEG string.

        Args:
            quality: JPEG compression quality (0-100)

        Returns:
            Base64-encoded JPEG string, or None if capture failed
        """
        frame = self.capture()
        if frame is None:
            return None

        return self.image_to_base64(frame, quality)

    @staticmethod
    def image_to_base64(image: np.ndarray, quality: int = 85) -> str:
        """
        Convert numpy image array to base64-encoded JPEG.

        Args:
            image: BGR image as numpy array
            quality: JPEG compression quality (0-100)

        Returns:
            Base64-encoded JPEG string
        """
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
        _, buffer = cv2.imencode('.jpg', image, encode_params)
        return base64.b64encode(buffer).decode('utf-8')

    def close(self):
        """Ensure camera is released (cleanup method)."""
        if self._cap:
            self._cap.release()
            self._cap = None
