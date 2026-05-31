from fastapi import FastAPI
from utils.face_recog import AttendanceSystem

system = AttendanceSystem()

app = FastAPI()


@app.post("/start")
def start_recognition_route():
    ok, message = system.start_recognition()
    return {
        "success": ok,
        "message": message
    }