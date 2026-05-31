import os
import cv2
import threading

class FrameFetcher:
    """
    Background thread that opens a fresh RTSP connection per frame,
    always keeping the single latest frame available for the inference thread.
    The inference thread never waits on network I/O — it just reads whatever
    the fetcher last stored.
    """
    def __init__(self, rtsp_url: str, retry_delay: float = 2.0):
        self._url = rtsp_url
        self._retry_delay = retry_delay

        self._frame = None         
        self._lock = threading.Lock()   
        self._running = True
        self._frame_ready = threading.Event()

        self._thread = threading.Thread(target=self._fetch_loop, daemon=True, name="FrameFetcher")
        self._thread.start()

    def _fetch_loop(self):
        cap = cv2.VideoCapture(self._url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        while self._running:
            ret, frame = cap.read()

            if ret and frame is not None:
                with self._lock:
                    self._frame = frame
                self._frame_ready.set()

    def get_frame(self, timeout: float = 10.0):
        """
        Block until a frame is available (only on cold start), then return
        a copy of the latest frame. Returns (True, frame) or (False, None).
        """
        if not self._frame_ready.wait(timeout=timeout):
            return False, None
        with self._lock:
            return True, self._frame.copy()

    def stop(self):
        self._running = False
        self._thread.join(timeout=3)