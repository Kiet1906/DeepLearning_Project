
import cv2
import time
from ultralytics import YOLO


def main():
    # Tai mo hinh YOLOv8 nano pretrained
    # Lan dau chay, Ultralytics co the tu tai file yolov8n.pt
    model = YOLO("yolov8n.pt")

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not cap.isOpened():
        cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("LOI: Khong mo duoc webcam.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("Nhan Q de thoat.")

    while True:
        start = time.perf_counter()

        success, frame = cap.read()
        if not success:
            print("Khong doc duoc frame.")
            break

        frame = cv2.flip(frame, 1)

        # Phat hien doi tuong tren frame
        results = model.predict(
            source=frame,
            imgsz=320,
            conf=0.4,
            verbose=False
        )

        # Ve khung phat hien
        output = results[0].plot()

        # Hien thi FPS xu ly thuc te
        elapsed = time.perf_counter() - start
        fps = 1.0 / elapsed if elapsed > 0 else 0

        cv2.putText(
            output,
            f"FPS: {fps:.1f}",
            (15, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        cv2.imshow("YOLOv8 Webcam Test", output)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
