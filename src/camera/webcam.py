
import cv2


class Webcam:
    def __init__(self, camera_index=0, width=640, height=480):
        self.cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)

        if not self.cap.isOpened():
            raise RuntimeError("Could not open webcam.")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def read(self):
        success, frame = self.cap.read()

        if not success:
            return None

        return cv2.flip(frame, 1)

    def release(self):
        self.cap.release()


if __name__ == "__main__":
    camera = Webcam()

    print("Webcam started. Press Q to exit.")

    try:
        while True:
            frame = camera.read()

            if frame is None:
                print("Failed to capture frame.")
                break

            cv2.imshow("SLMCI - Webcam Test", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        camera.release()
        cv2.destroyAllWindows()
