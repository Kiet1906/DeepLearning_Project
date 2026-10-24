
import cv2
import mediapipe as mp


def main():
    model_path = "face_landmarker.task"

    # Cau hinh Face Landmarker
    base_options = mp.tasks.BaseOptions(
        model_asset_path=model_path
    )

    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False
    )

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not cap.isOpened():
        cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("LOI: Khong mo duoc webcam.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("Nhan Q de thoat.")

    try:
        with mp.tasks.vision.FaceLandmarker.create_from_options(
            options
        ) as landmarker:

            while True:
                success, frame = cap.read()

                if not success:
                    print("Khong doc duoc frame.")
                    break

                frame = cv2.flip(frame, 1)

                rgb_frame = cv2.cvtColor(
                    frame, cv2.COLOR_BGR2RGB
                )

                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb_frame
                )

                # VIDEO mode yeu cau timestamp tang dan
                timestamp_ms = int(
                    cap.get(cv2.CAP_PROP_POS_MSEC)
                )

                # Webcam thuong khong cung cap timestamp hop ly,
                # nen tu tao timestamp tang dan
                if not hasattr(main, "timestamp_ms"):
                    main.timestamp_ms = 0
                else:
                    main.timestamp_ms += 1

                result = landmarker.detect_for_video(
                    mp_image,
                    main.timestamp_ms
                )

                if result.face_landmarks:
                    h, w, _ = frame.shape

                    for face in result.face_landmarks:
                        for point in face:
                            x = int(point.x * w)
                            y = int(point.y * h)

                            if 0 <= x < w and 0 <= y < h:
                                cv2.circle(
                                    frame,
                                    (x, y),
                                    1,
                                    (0, 255, 0),
                                    -1
                                )

                    cv2.putText(
                        frame,
                        "Face detected",
                        (15, 35),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        2
                    )
                else:
                    cv2.putText(
                        frame,
                        "No face detected",
                        (15, 35),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 0, 255),
                        2
                    )

                cv2.imshow("MediaPipe Face Landmarker", frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    except Exception as error:
        print("Loi MediaPipe:", error)

    finally:
        cap.release()
        cv2.destroyAllWindows()

    print("Da dong webcam.")


if __name__ == "__main__":
    main()
