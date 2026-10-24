
import cv2
import mediapipe as mp
import math
import time
import numpy as np


# Cac diem moc quanh mieng
# 13, 14: moi tren va moi duoi
# 78, 308: hai ben mieng
# 82, 312: cac diem phu de do do mo mieng
MOUTH_TOP = 13
MOUTH_BOTTOM = 14
MOUTH_LEFT = 78
MOUTH_RIGHT = 308

# Nguong MAR thu nghiem, se dieu chinh sau khi kiem tra
MAR_THRESHOLD = 0.35

# Thoi gian mieng mo lien tuc de danh dau
MOUTH_OPEN_DURATION = 0.8


def distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def calculate_mar(landmarks, width, height):
    top = landmarks[MOUTH_TOP]
    bottom = landmarks[MOUTH_BOTTOM]
    left = landmarks[MOUTH_LEFT]
    right = landmarks[MOUTH_RIGHT]

    top_point = (int(top.x * width), int(top.y * height))
    bottom_point = (int(bottom.x * width), int(bottom.y * height))
    left_point = (int(left.x * width), int(left.y * height))
    right_point = (int(right.x * width), int(right.y * height))

    vertical_distance = distance(top_point, bottom_point)
    horizontal_distance = distance(left_point, right_point)

    if horizontal_distance == 0:
        return 0.0, [
            top_point, bottom_point, left_point, right_point
        ]

    mar = vertical_distance / horizontal_distance

    return mar, [
        top_point, bottom_point, left_point, right_point
    ]


def main():
    mp_tasks = mp.tasks

    options = mp_tasks.vision.FaceLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(
            model_asset_path="face_landmarker.task"
        ),
        running_mode=mp_tasks.vision.RunningMode.VIDEO,
        num_faces=1
    )

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not cap.isOpened():
        cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("LOI: Khong mo duoc webcam.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    mouth_open_start = None
    start_time = time.perf_counter()
    last_timestamp_ms = -1

    try:
        with mp_tasks.vision.FaceLandmarker.create_from_options(
            options
        ) as landmarker:

            while True:
                success, frame = cap.read()

                if not success:
                    print("Khong doc duoc frame.")
                    break

                frame = cv2.flip(frame, 1)
                height, width, _ = frame.shape

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb
                )

                timestamp_ms = int(
                    (time.perf_counter() - start_time) * 1000
                )
                timestamp_ms = max(
                    timestamp_ms, last_timestamp_ms + 1
                )
                last_timestamp_ms = timestamp_ms

                result = landmarker.detect_for_video(
                    mp_image, timestamp_ms
                )

                if result.face_landmarks:
                    landmarks = result.face_landmarks[0]

                    mar, mouth_points = calculate_mar(
                        landmarks, width, height
                    )

                    now = time.perf_counter()

                    if mar >= MAR_THRESHOLD:
                        if mouth_open_start is None:
                            mouth_open_start = now
                    else:
                        mouth_open_start = None

                    open_duration = (
                        now - mouth_open_start
                        if mouth_open_start is not None
                        else 0.0
                    )

                    mouth_open = mar >= MAR_THRESHOLD

                    for point in mouth_points:
                        cv2.circle(
                            frame, point, 4, (0, 255, 255), -1
                        )

                    cv2.line(
                        frame, mouth_points[0], mouth_points[1],
                        (0, 255, 0), 2
                    )
                    cv2.line(
                        frame, mouth_points[2], mouth_points[3],
                        (255, 0, 0), 2
                    )

                    if mouth_open and open_duration >= MOUTH_OPEN_DURATION:
                        status = "MOUTH OPEN - LONG"
                        color = (0, 0, 255)
                    elif mouth_open:
                        status = "MOUTH OPEN"
                        color = (0, 165, 255)
                    else:
                        status = "MOUTH CLOSED"
                        color = (0, 255, 0)

                    cv2.putText(
                        frame,
                        f"MAR: {mar:.3f} (Threshold: {MAR_THRESHOLD})",
                        (15, 35),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 0),
                        2
                    )

                    cv2.putText(
                        frame,
                        status,
                        (15, 70),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        color,
                        2
                    )

                    cv2.putText(
                        frame,
                        f"Mouth open duration: {open_duration:.1f}s",
                        (15, 105),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 0),
                        2
                    )

                else:
                    mouth_open_start = None

                    cv2.putText(
                        frame,
                        "NO FACE",
                        (15, 35),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 0, 255),
                        2
                    )

                cv2.imshow("Mouth MAR Detection Test", frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    except Exception as error:
        print("Loi:", error)

    finally:
        cap.release()
        cv2.destroyAllWindows()

    print("Da dong webcam.")


if __name__ == "__main__":
    main()
