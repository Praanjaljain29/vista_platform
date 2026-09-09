from flask import Flask, render_template, request, send_from_directory
from werkzeug.utils import secure_filename
from ultralytics import YOLO
import cv2
import csv
import os
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
RESULT_DIR = os.path.join(BASE_DIR, "results")
MODEL_PATH = os.path.join(BASE_DIR, "yolo26n.pt")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

app = Flask(__name__)

# Maximum upload size = 500 MB
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024

ALLOWED_EXTENSIONS = {
    "mp4",
    "avi",
    "mov",
    "mkv",
    "webm"
}

CONFIDENCE = 0.35

print("Loading YOLO26 model...")
model = YOLO(MODEL_PATH)


def allowed_file(filename: str) -> bool:
    """
    Check whether the uploaded file has an allowed video extension.
    """
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


def process_video(video_path: str, query: str, job_id: str):
    """
    Run YOLO26 + ByteTrack on the uploaded video.

    Saves:
    1. Complete detection CSV
    2. Classes CSV
    3. Top 2 matching object crops
    """

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video: {video_path}"
        )

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    )

    detections_csv = os.path.join(
        RESULT_DIR,
        f"{job_id}_detections.csv"
    )

    classes_csv = os.path.join(
        RESULT_DIR,
        f"{job_id}_classes.csv"
    )

    # Simple query for now.
    # Examples: person, car, dog
    target = query.strip().lower()

    # Store only the top two highest-confidence matches
    best_matches = []

    # Keep track of unique classes
    classes_seen = {}

    frame_no = 0
    serial_no = 1

    start_time = time.time()

    # --------------------------------
    # CREATE DETECTIONS CSV
    # --------------------------------

    with open(
        detections_csv,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "serial_no",
            "frame_no",
            "timestamp",
            "object_class",
            "class_id",
            "track_id",
            "confidence",
            "x1",
            "y1",
            "x2",
            "y2"
        ])

        # --------------------------------
        # PROCESS VIDEO
        # --------------------------------

        while True:

            success, frame = cap.read()

            if not success:
                break

            frame_no += 1

            # YOLO26 + ByteTrack
            results = model.track(
                frame,
                persist=True,
                tracker="bytetrack.yaml",
                conf=CONFIDENCE,
                verbose=False
            )

            for result in results:

                if (
                    result.boxes is None
                    or result.boxes.id is None
                ):
                    continue

                boxes = result.boxes

                for i in range(len(boxes)):

                    # Class
                    class_id = int(boxes.cls[i])
                    class_name = result.names[class_id]

                    # Confidence
                    confidence = float(
                        boxes.conf[i]
                    )

                    # Tracking ID
                    track_id = int(
                        boxes.id[i]
                    )

                    # Bounding box
                    x1, y1, x2, y2 = (
                        boxes.xyxy[i].tolist()
                    )

                    # Timestamp
                    timestamp = frame_no / fps

                    # Store unique class
                    classes_seen[class_id] = class_name

                    # --------------------------------
                    # WRITE DETECTION TO CSV
                    # --------------------------------

                    writer.writerow([
                        serial_no,
                        frame_no,
                        round(timestamp, 3),
                        class_name,
                        class_id,
                        track_id,
                        round(confidence, 4),
                        round(x1, 2),
                        round(y1, 2),
                        round(x2, 2),
                        round(y2, 2)
                    ])

                    serial_no += 1

                    # --------------------------------
                    # CHECK QUERY MATCH
                    # --------------------------------

                    if (
                        target
                        and target in class_name.lower()
                    ):

                        # Small padding around crop
                        pad = 10

                        h, w = frame.shape[:2]

                        ix1 = max(
                            0,
                            int(x1) - pad
                        )

                        iy1 = max(
                            0,
                            int(y1) - pad
                        )

                        ix2 = min(
                            w,
                            int(x2) + pad
                        )

                        iy2 = min(
                            h,
                            int(y2) + pad
                        )

                        crop = frame[
                            iy1:iy2,
                            ix1:ix2
                        ]

                        if crop.size > 0:

                            candidate = {
                                "confidence": confidence,
                                "frame_no": frame_no,
                                "class_name": class_name,
                                "track_id": track_id,
                                "crop": crop.copy()
                            }

                            best_matches.append(
                                candidate
                            )

                            # Highest confidence first
                            best_matches.sort(
                                key=lambda x: x["confidence"],
                                reverse=True
                            )

                            # Keep only top 2
                            best_matches = best_matches[:2]

    cap.release()

    # --------------------------------
    # SAVE TOP 2 IMAGES
    # --------------------------------

    output_images = []

    for rank, item in enumerate(
        best_matches,
        start=1
    ):

        filename = (
            f"{job_id}_top_{rank}.jpg"
        )

        image_path = os.path.join(
            RESULT_DIR,
            filename
        )

        cv2.imwrite(
            image_path,
            item["crop"]
        )

        output_images.append({
            "filename": filename,
            "frame_no": item["frame_no"],
            "class_name": item["class_name"],
            "confidence": round(
                item["confidence"],
                3
            ),
            "track_id": item["track_id"]
        })

    # --------------------------------
    # WRITE CLASSES CSV
    # --------------------------------

    with open(
        classes_csv,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "class_id",
            "object_class"
        ])

        for class_id in sorted(classes_seen):

            writer.writerow([
                class_id,
                classes_seen[class_id]
            ])

    elapsed = time.time() - start_time

    return {
        "frames_processed": frame_no,
        "detections": serial_no - 1,
        "unique_classes": len(classes_seen),
        "time_minutes": round(
            elapsed / 60,
            2
        ),
        "images": output_images,
        "detections_csv": os.path.basename(
            detections_csv
        ),
        "classes_csv": os.path.basename(
            classes_csv
        )
    }


# ======================================
# HOME PAGE
# ======================================

@app.route("/", methods=["GET"])
def index():

    return render_template(
        "index.html",
        result=None,
        error=None
    )


# ======================================
# PROCESS VIDEO
# ======================================

@app.route(
    "/process",
    methods=["POST"]
)
def process():

    video_file = request.files.get(
        "video"
    )

    query = request.form.get(
        "query",
        ""
    ).strip()

    # Check video
    if (
        not video_file
        or video_file.filename == ""
    ):

        return render_template(
            "index.html",
            result=None,
            error="Please choose a video."
        )

    # Check extension
    if not allowed_file(
        video_file.filename
    ):

        return render_template(
            "index.html",
            result=None,
            error=(
                "Please upload an MP4, AVI, "
                "MOV, MKV, or WEBM video."
            )
        )

    # Check query
    if not query:

        return render_template(
            "index.html",
            result=None,
            error=(
                "Enter an object class, "
                "e.g. person or car."
            )
        )

    filename = secure_filename(
        video_file.filename
    )

    job_id = (
        f"job_{int(time.time())}"
    )

    video_path = os.path.join(
        UPLOAD_DIR,
        f"{job_id}_{filename}"
    )

    # Save uploaded video
    video_file.save(video_path)

    try:

        result = process_video(
            video_path,
            query,
            job_id
        )

        return render_template(
            "index.html",
            result=result,
            error=None,
            query=query
        )

    except Exception as exc:

        return render_template(
            "index.html",
            result=None,
            error=str(exc),
            query=query
        )


# ======================================
# SERVE RESULT IMAGES
# ======================================

@app.route(
    "/results/<path:filename>"
)
def result_file(filename):

    return send_from_directory(
        RESULT_DIR,
        filename,
        as_attachment=False
    )


# ======================================
# DOWNLOAD CSV
# ======================================

@app.route(
    "/download/<path:filename>"
)
def download_file(filename):

    return send_from_directory(
        RESULT_DIR,
        filename,
        as_attachment=True
    )


# ======================================
# START FLASK
# ======================================

if __name__ == "__main__":

    app.run(
        debug=True
    )