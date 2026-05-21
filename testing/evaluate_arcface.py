import onnxruntime as ort
import os
import cv2
import pickle
import numpy as np
from insightface.app import FaceAnalysis
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

RECOGNITION_THRESHOLD = 0.65
DB_PATH               = "attendance_db.pkl"
VIDEOS_DIR            = "eval_videos"
NUM_WORKERS           = 5

_thread_local = threading.local()

def get_app():
    if not hasattr(_thread_local, "app"):
        _thread_local.app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
        _thread_local.app.prepare(ctx_id=0, det_size=(640, 640))
    return _thread_local.app

def load_db(db_path):
    known_faces = {}
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database not found: {db_path}")
    with open(db_path, "rb") as f:
        while True:
            try:
                data = pickle.load(f)
                if isinstance(data, dict):
                    known_faces.update(data)
            except EOFError:
                break
    return known_faces

def cosine_similarity(emb1, emb2):
    return float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)))

def predict_frame(frame, known_faces):
    faces = get_app().get(frame)
    results = []
    for face in faces:
        best_entry, best_sim = None, 0.0
        for entry_no, data in known_faces.items():
            sim = cosine_similarity(face.normed_embedding, data["embedding"])
            if sim > best_sim:
                best_sim = sim
                best_entry = entry_no
        results.append({
            "predicted_entry": best_entry if best_sim >= RECOGNITION_THRESHOLD else None,
            "best_entry":      best_entry,   # always the closest match regardless of threshold
            "similarity":      best_sim,
            "above_threshold": best_sim >= RECOGNITION_THRESHOLD,
            "det_score":       float(face.det_score),
        })
    return results

def evaluate_video(known_faces, video_path, true_entry):
    is_impostor = (true_entry is None)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  [WARN] Cannot open {video_path}")
        return None

    total_frames     = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps              = cap.get(cv2.CAP_PROP_FPS)
    frames_processed = 0
    frames_with_face = 0
    frames_no_face   = 0

    frames_correct  = 0
    frames_wrong_id = 0
    frames_no_match = 0

    frames_correctly_rejected = 0
    frames_falsely_accepted   = 0

    all_sims        = []   # every face's best similarity, every frame
    true_entry_sims = []   # sims where prediction == true_entry
    wrong_id_sims   = []   # sims where above threshold but wrong identity
    no_match_sims   = []   # sims where below threshold
    impostor_sims   = []   # for impostors: all best-match sims

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames_processed += 1

        preds = predict_frame(frame, known_faces)
        if not preds:
            frames_no_face += 1
            continue

        frames_with_face += 1
        for p in preds:
            all_sims.append(p["similarity"])

            if is_impostor:
                impostor_sims.append(p["similarity"])
                if p["above_threshold"]:
                    frames_falsely_accepted += 1
                else:
                    frames_correctly_rejected += 1
            else:
                if p["predicted_entry"] == true_entry:
                    frames_correct += 1
                    true_entry_sims.append(p["similarity"])
                elif p["above_threshold"]:
                    frames_wrong_id += 1
                    wrong_id_sims.append(p["similarity"])
                else:
                    frames_no_match += 1
                    no_match_sims.append(p["similarity"])

    cap.release()

    if frames_processed == 0:
        return None

    d = frames_with_face or 1

    base = {
        "video":            os.path.basename(video_path),
        "true_entry":       true_entry if not is_impostor else "IMPOSTOR",
        "is_impostor":      is_impostor,
        "total_frames":     total_frames,
        "frames_processed": frames_processed,
        "frames_with_face": frames_with_face,
        "frames_no_face":   frames_no_face,
        "fps":              fps,
        "duration_s":       total_frames / fps if fps > 0 else 0,
    }

    if is_impostor:
        base.update({
            "frames_correctly_rejected": frames_correctly_rejected,
            "frames_falsely_accepted":   frames_falsely_accepted,
            "true_negative_rate":        frames_correctly_rejected / d,
            "false_accept_rate":         frames_falsely_accepted   / d,
            # Closest anyone in DB ever got to this impostor
            "max_sim_to_anyone":  float(np.max(impostor_sims))  if impostor_sims else 0.0,
            "mean_sim_to_anyone": float(np.mean(impostor_sims)) if impostor_sims else 0.0,
        })
    else:
        base.update({
            "frames_correct":    frames_correct,
            "frames_wrong_id":   frames_wrong_id,
            "frames_no_match":   frames_no_match,
            "true_accept_rate":  frames_correct  / d,
            "false_reject_rate": frames_no_match / d,
            "false_accept_rate": frames_wrong_id / d,
            # Correct-match similarity stats
            "mean_sim_correct":  float(np.mean(true_entry_sims)) if true_entry_sims else 0.0,
            "min_sim_correct":   float(np.min(true_entry_sims))  if true_entry_sims else 0.0,
            "max_sim_correct":   float(np.max(true_entry_sims))  if true_entry_sims else 0.0,
            # Below-threshold frames: how far below were they?
            "mean_sim_missed":   float(np.mean(no_match_sims))   if no_match_sims   else 0.0,
            "max_sim_missed":    float(np.max(no_match_sims))    if no_match_sims   else 0.0,
        })

    return base

def worker(known_faces, video_path, true_entry):
    vf    = os.path.basename(video_path)
    label = true_entry if true_entry else "IMPOSTOR"
    print(f"  [START] {vf}  ({label})")
    result = evaluate_video(known_faces, video_path, true_entry)
    if result:
        if result["is_impostor"]:
            print(f"  [DONE]  {vf}  TNR={result['true_negative_rate']:.1%}  "
                  f"MaxSim={result['max_sim_to_anyone']:.4f}")
        else:
            print(f"  [DONE]  {vf}  TAR={result['true_accept_rate']:.1%}  "
                  f"FRR={result['false_reject_rate']:.1%}  "
                  f"MeanSim={result['mean_sim_correct']:.4f}")
    return result

def print_report(results):
    known    = [r for r in results if r and not r["is_impostor"]]
    impostor = [r for r in results if r and r["is_impostor"]]

    S1 = "─" * 130
    S2 = "═" * 130

    # ── Known persons ──────────────────────────────────────────────────
    print(f"\n{S2}")
    print(f"{'KNOWN PERSONS':^130}")
    print(f"{S2}")
    print(f"{'Video':<25} {'Entry':<14} {'Dur':>5} {'Frames':>7} {'Faces':>6} "
          f"{'NoFace':>7} {'TAR':>6} {'FRR':>6} {'FAR':>6} "
          f"{'MeanOK':>7} {'MinOK':>7} {'MaxOK':>7} "
          f"{'MeanMiss':>9} {'MaxMiss':>8}  Result")
    print(S1)

    for r in sorted(known, key=lambda x: x["video"]):
        status = "✓ PASS" if r["true_accept_rate"] >= 0.5 else "✗ FAIL"
        dur    = f"{r['duration_s']:.0f}s"
        print(
            f"{r['video']:<25} {r['true_entry']:<14} {dur:>5} {r['total_frames']:>7} "
            f"{r['frames_with_face']:>6} {r['frames_no_face']:>7} "
            f"{r['true_accept_rate']:>6.1%} {r['false_reject_rate']:>6.1%} {r['false_accept_rate']:>6.1%} "
            f"{r['mean_sim_correct']:>7.4f} {r['min_sim_correct']:>7.4f} {r['max_sim_correct']:>7.4f} "
            f"{r['mean_sim_missed']:>9.4f} {r['max_sim_missed']:>8.4f}  {status}"
        )

    if known:
        print(S1)
        print(f"  Mean TAR={np.mean([r['true_accept_rate']  for r in known]):.2%}  "
              f"Mean FRR={np.mean([r['false_reject_rate'] for r in known]):.2%}  "
              f"Mean FAR={np.mean([r['false_accept_rate'] for r in known]):.2%}  "
              f"Pass: {sum(r['true_accept_rate']>=0.5 for r in known)}/{len(known)}")

    # ── Impostors ──────────────────────────────────────────────────────
    if impostor:
        print(f"\n{S2}")
        print(f"{'IMPOSTOR / UNKNOWN PERSONS':^130}")
        print(f"{S2}")
        print(f"{'Video':<25} {'Dur':>5} {'Frames':>7} {'Faces':>6} {'NoFace':>7} "
              f"{'TNR':>7} {'FAR':>6} "
              f"{'MaxSim':>8} {'MeanSim':>9}  Result")
        print(S1)

        for r in sorted(impostor, key=lambda x: x["video"]):
            status = "✓ PASS" if r["false_accept_rate"] <= 0.05 else "✗ FAIL"
            dur    = f"{r['duration_s']:.0f}s"
            print(
                f"{r['video']:<25} {dur:>5} {r['total_frames']:>7} "
                f"{r['frames_with_face']:>6} {r['frames_no_face']:>7} "
                f"{r['true_negative_rate']:>7.1%} {r['false_accept_rate']:>6.1%} "
                f"{r['max_sim_to_anyone']:>8.4f} {r['mean_sim_to_anyone']:>9.4f}  {status}"
            )

        print(S1)
        print(f"  Mean TNR={np.mean([r['true_negative_rate'] for r in impostor]):.2%}  "
              f"Mean FAR={np.mean([r['false_accept_rate']   for r in impostor]):.2%}  "
              f"Pass: {sum(r['false_accept_rate']<=0.05 for r in impostor)}/{len(impostor)}")

    print(f"\n{S2}\n")

def main():
    print("Loading database...")
    known_faces = load_db(DB_PATH)
    print(f"  {len(known_faces)} registered: {list(known_faces.keys())}")

    video_files = [
        f for f in os.listdir(VIDEOS_DIR)
        if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))
    ]
    if not video_files:
        print(f"No videos found in {VIDEOS_DIR}")
        return

    jobs = []
    for vf in sorted(video_files):
        stem       = os.path.splitext(vf)[0]
        true_entry = stem if stem in known_faces else None
        print(f"  Queued: {vf} → {true_entry if true_entry else 'IMPOSTOR'}")
        jobs.append((os.path.join(VIDEOS_DIR, vf), true_entry))

    print(f"\nRunning {len(jobs)} videos on {NUM_WORKERS} threads...\n")

    results = []
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as pool:
        futures = {
            pool.submit(worker, known_faces, path, entry): path
            for path, entry in jobs
        }
        for future in as_completed(futures):
            result = future.result()
            if result:
                results.append(result)

    print_report(results)

if __name__ == "__main__":
    main()