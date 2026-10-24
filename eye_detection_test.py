
import cv2
import mediapipe as mp
import math
import time
import numpy as np
import winsound
import threading


# Cac diem moc quanh mat
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

# Nguong EAR
EAR_THRESHOLD = 0.175

# Thoi gian nham mat lien tuc de canh bao
CLOSED_EYE_DURATION = 1.0

# Khoang cach giua cac lan phat am thanh (giay)
ALARM_INTERVAL = 1.5


def distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def calculate_ear(landmarks, eye_indices, width, height):
    points = []

    for index in eye_indices:
        landmark = landmarks[index]
        x = int(landmark.x * width)
        y = int(landmark.y * height)
        points.append((x, y))

    vertical_1 = distance(points[1], points[5])
    vertical_2 = distance(points[2], points[4])
    horizontal = distance(points[0], points[3])

    if horizontal == 0:
        return 0.0, points

    ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
    return ear, points


def play_alarm():
    try:
        winsound.Beep(1500, 500)
    except RuntimeError:
        winsound.MessageBeep(winsound.MB_ICONEXCLAMATION)


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

    closed_start = None
    last_alarm_time = 0.0
    start_time = time.perf_counter()
    last_timestamp_ms = -1
    alarm_active = False

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

                    left_ear, left_points = calculate_ear(
                        landmarks, LEFT_EYE, width, height
                    )
                    right_ear, right_points = calculate_ear(
                        landmarks, RIGHT_EYE, width, height
                    )

                    ear = (left_ear + right_ear) / 2.0
                    now = time.perf_counter()

                    if ear < EAR_THRESHOLD:
                        if closed_start is None:
                            closed_start = now
                    else:
                        closed_start = None
                        alarm_active = False

                    closed_duration = (
                        now - closed_start
                        if closed_start is not None
                        else 0.0
                    )

                    eyes_closed = ear < EAR_THRESHOLD
                    prolonged_closure = (
                        closed_duration >= CLOSED_EYE_DURATION
                    )

                    for points in (left_points, right_points):
                        for point in points:
                            cv2.circle(
                                frame, point, 2, (0, 255, 255), -1
                            )

                        cv2.polylines(
                            frame,
                            [np.array(points, dtype=np.int32)],
                            True,
                            (0, 255, 255),
                            1
                        )

                    if prolonged_closure:
                        status = "EYES CLOSED - LONG"
                        color = (0, 0, 255)

                        if (
                            not alarm_active
                            and now - last_alarm_time >= ALARM_INTERVAL
                        ):
                            threading.Thread(
                                target=play_alarm,
                                daemon=True
                            ).start()
                            last_alarm_time = now
                            alarm_active = True

                    elif eyes_closed:
                        status = "EYES CLOSED"
                        color = (0, 165, 255)
                    else:
                        status = "EYES OPEN"
                        color = (0, 255, 0)

                    cv2.putText(
                        frame,
                        f"EAR: {ear:.3f} (Threshold: {EAR_THRESHOLD})",
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
                        f"Closed duration: {closed_duration:.1f}s",
                        (15, 105),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 0),
                        2
                    )

                    if prolonged_closure:
                        cv2.putText(
                            frame,
                            "WARNING: PLEASE WAKE UP!",
                            (15, 145),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.65,
                            (0, 0, 255),
                            2
                        )

                else:
                    closed_start = None
                    alarm_active = False

                    cv2.putText(
                        frame,
                        "NO FACE",
                        (15, 35),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 0, 255),
                        2
                    )

                cv2.imshow("Driver Drowsiness Detection", frame)

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
