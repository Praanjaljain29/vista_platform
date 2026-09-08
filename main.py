from ultralytics import YOLO
import cv2
import csv
import time


# =========================
# SETTINGS
# =========================

VIDEO_PATH = "kitchen.mp4"

DETECTIONS_CSV = "detections.csv"
CLASSES_CSV = "classes.csv"

CONFIDENCE = 0.35


# =========================
# LOAD MODEL + VIDEO
# =========================

print("Loading YOLO model...")
model = YOLO("yolo26n.pt")

video = cv2.VideoCapture(VIDEO_PATH)

if not video.isOpened():
    raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

fps = video.get(cv2.CAP_PROP_FPS)
total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"FPS: {fps}")
print(f"Total frames: {total_frames}")
print(f"Duration: {total_frames / fps:.2f} seconds")
print("Starting detection + tracking...\n")


# =========================
# CSV SETUP
# =========================

detections_file = open(
    DETECTIONS_CSV,
    "w",
    newline="",
    encoding="utf-8"
)

writer = csv.writer(detections_file)

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


# Store unique classes
classes_seen = {}


# =========================
# PROCESS VIDEO
# =========================

frame_no = 0
serial_no = 1

start_time = time.time()

while True:

    success, frame = video.read()

    if not success:
        break

    # YOLO detection + ByteTrack tracking
    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        conf=CONFIDENCE,
        verbose=False
    )

    for result in results:

        if result.boxes is None:
            continue

        boxes = result.boxes

        # No tracking IDs assigned
        if boxes.id is None:
            continue

        for i in range(len(boxes)):

            class_id = int(boxes.cls[i])
            class_name = result.names[class_id]

            confidence = float(boxes.conf[i])

            track_id = int(boxes.id[i])

            # Bounding box
            x1, y1, x2, y2 = boxes.xyxy[i].tolist()

            # Timestamp in seconds
            timestamp = frame_no / fps

            # Save class
            classes_seen[class_id] = class_name

            # Write detection
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

    frame_no += 1

    # Progress every 300 frames
    if frame_no % 300 == 0:

        elapsed = time.time() - start_time

        progress = (frame_no / total_frames) * 100

        print(
            f"Processed {frame_no}/{total_frames} "
            f"frames ({progress:.1f}%) | "
            f"Detections: {serial_no - 1} | "
            f"Time: {elapsed:.1f}s"
        )


# =========================
# CLEAN UP
# =========================

video.release()
detections_file.close()


# =========================
# WRITE CLASSES CSV
# =========================

with open(
    CLASSES_CSV,
    "w",
    newline="",
    encoding="utf-8"
) as classes_file:

    class_writer = csv.writer(classes_file)

    class_writer.writerow([
        "class_id",
        "object_class"
    ])

    for class_id in sorted(classes_seen):
        class_writer.writerow([
            class_id,
            classes_seen[class_id]
        ])


elapsed = time.time() - start_time

print("\n==============================")
print("PROCESSING COMPLETE")
print("==============================")
print(f"Frames processed: {frame_no}")
print(f"Total detections: {serial_no - 1}")
print(f"Unique classes: {len(classes_seen)}")
print(f"Time taken: {elapsed / 60:.2f} minutes")
print(f"\nCreated:")
print(f"  {DETECTIONS_CSV}")
print(f"  {CLASSES_CSV}")