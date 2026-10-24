
import cv2
import time


def main():
    # Thu mo webcam bang DirectShow tren Windows
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print("DirectShow khong mo duoc webcam, thu backend mac dinh...")
        cap.release()
        cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("LOI: Khong mo duoc webcam.")
        return

    # Giam kich thuoc de giam tai xu ly
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    print("Camera:", int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
          "x", int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    print("Backend:", cap.getBackendName())
    print("Nhan Q de thoat.")

    frame_count = 0
    start_time = time.perf_counter()
    fps = 0.0

    while True:
        success, frame = cap.read()

        if not success:
            print("Khong doc duoc frame.")
            break

        frame = cv2.flip(frame, 1)
        frame_count += 1

        # Tinh FPS trung binh moi 1 giay
        elapsed = time.perf_counter() - start_time

        if elapsed >= 1.0:
            fps = frame_count / elapsed
            frame_count = 0
            start_time = time.perf_counter()

        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (15, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            "Press Q to quit",
            (15, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2
        )

        cv2.imshow("Driver Drowsiness - Webcam Test", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Da dong webcam.")


if __name__ == "__main__":
    main()
