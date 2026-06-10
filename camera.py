import os
import time
from datetime import datetime
import cv2
from onvif import ONVIFCamera
from zeep.transports import Transport
from requests import Session
import urllib3

# disable warnings (optional but cleaner)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# create session with SSL disabled
session = Session()
session.verify = False

transport = Transport(session=session)

# ----------------------------
# Camera config
# ----------------------------
HOST = "10.208.22.128"
PORT = 80
USERNAME = "admin"
PASSWORD = "SOumil@@btp1"

RTSP_URL = f"rtsp://admin:SOumil%40%40btp1@{HOST}:554/video/live?channel=1&subtype=0"

SCREENSHOT_DIR = "screenshots"

KEY_SCREENSHOT = ord("p")
KEY_QUIT = ord("e")

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def choose_profile(media):
    profiles = media.GetProfiles()
    if not profiles:
        raise RuntimeError("No media profiles found")

    return profiles[0]


def get_rtsp_url() -> str:
    return RTSP_URL

def save_screenshot(frame, out_dir=SCREENSHOT_DIR):
    ensure_dir(out_dir)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = os.path.join(out_dir, f"frame_{ts}.jpg")
    ok = cv2.imwrite(filename, frame)
    if ok:
        print(f"[OK] Saved screenshot: {filename}")
    else:
        print("[WARN] Failed to save screenshot")

def draw_overlay(frame, lines):
    y = 25
    for line in lines:
        cv2.putText(
            frame,
            line,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        y += 25
    return frame


    
def main():
    cam = ONVIFCamera(HOST, PORT, USERNAME, PASSWORD, transport=transport)
    print(f"Connected to camera at {HOST}:{PORT}")
    dev = cam.create_devicemgmt_service()

    print(dev.GetDeviceInformation())

    rtsp_url = get_rtsp_url()
    print(f"RTSP URL: {rtsp_url}")

    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open stream: {rtsp_url}")

    print("\nControls:")
    print("  P        -> Save screenshot")
    print("  E        -> Exit\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("[WARN] Failed to grab frame")
                time.sleep(0.1)
                continue


            overlay_lines = [
                "P: Screenshot  E: Exit",
            ]
            frame = draw_overlay(frame, overlay_lines)

            cv2.imshow("CP PLUS Camera", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == KEY_QUIT:
                break
            elif key == KEY_SCREENSHOT:
                save_screenshot(frame)

    finally:

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    print("Starting camera viewer...")
    main()