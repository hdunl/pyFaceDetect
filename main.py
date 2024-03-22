import cv2
import dlib
import numpy as np

predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
detector = dlib.get_frontal_face_detector()
video_capture = cv2.VideoCapture(0)

color_ranges = {
    "white": [(0, 0, 0), (255, 145, 190)],
    "tan": [(0, 0, 0), (255, 140, 215)],
    "brown": [(0, 0, 0), (255, 130, 180)],
    "black": [(0, 0, 0), (255, 115, 150)],
    "green": [(0, 125, 0), (255, 255, 255)],
    "yellow": [(0, 135, 120), (255, 255, 255)],
}

tracker = None
face_color = "Unknown"


def detect_face_color(face_region):
    global face_color
    lab_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2Lab)
    avg_color = np.mean(lab_face, axis=(0, 1)).astype(int)
    for color, (lower, upper) in color_ranges.items():
        if np.all(avg_color >= lower) and np.all(avg_color <= upper):
            face_color = color
            break
    return lab_face, avg_color


while True:
    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if tracker is not None:
        tracking_quality = tracker.update(gray)
        if tracking_quality >= 7:
            pos = tracker.get_position()
            x1, y1, x2, y2 = int(pos.left()), int(pos.top()), int(pos.right()), int(pos.bottom())
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(frame.shape[1], x2)
            y2 = min(frame.shape[0], y2)
            face_region = frame[y1:y2, x1:x2]

            if face_region.size > 0:
                lab_face, avg_color = detect_face_color(face_region)
                cv2.putText(frame, f"Face Color: {face_color}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                shape = predictor(gray, dlib.rectangle(x1, y1, x2, y2))
                landmarks = np.array([(shape.part(i).x, shape.part(i).y) for i in range(68)], dtype=np.int32)

                for i, j in [(0, 16), (17, 21), (22, 26), (27, 30), (31, 35), (36, 41), (42, 47), (48, 59), (60, 67)]:
                    pts = landmarks[i:j + 1]
                    for k in range(1, len(pts)):
                        cv2.line(frame, tuple(pts[k - 1]), tuple(pts[k]), (0, 255, 0), 1)

                cv2.polylines(frame, [landmarks[0:17]], False, (0, 255, 0), 1)
                cv2.polylines(frame, [landmarks[17:22]], False, (0, 255, 0), 1)
                cv2.polylines(frame, [landmarks[22:27]], False, (0, 255, 0), 1)
                cv2.polylines(frame, [landmarks[27:31]], False, (0, 255, 0), 1)
                cv2.polylines(frame, [landmarks[31:36]], False, (0, 255, 0), 1)

                cv2.imshow("Face Region", face_region)
                cv2.imshow("LAB Color Space", lab_face)

                lab_info = f"L: {avg_color[0]}, A: {avg_color[1]}, B: {avg_color[2]}"
                cv2.putText(frame, lab_info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Tracking Quality: {tracking_quality}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                face_landmarks = np.zeros_like(frame)
                for pt in landmarks:
                    cv2.circle(face_landmarks, tuple(pt), 2, (0, 255, 0), -1)
                cv2.imshow("Face Landmarks", face_landmarks)

                binary_face_region = \
                    cv2.threshold(cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY), 60, 255, cv2.THRESH_BINARY)[1]
                cv2.imshow("Binary Face Region", binary_face_region)

                canny_edges = cv2.Canny(face_region, 100, 200)
                cv2.imshow("Canny Edges", canny_edges)

                face_histogram = cv2.calcHist([face_region], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
                cv2.normalize(face_histogram, face_histogram, 0, 255, cv2.NORM_MINMAX)
                hist_image = np.zeros((300, 256, 3), dtype=np.uint8)
                for i in range(8):
                    for j in range(8):
                        for k in range(8):
                            cv2.line(hist_image, (int(face_histogram[i, j, k]), 299),
                                     (int(face_histogram[i, j, k]), 300),
                                     (255, 0, 0), thickness=1)
                cv2.imshow("Face Color Histogram", hist_image)
        else:
            tracker = None
            face_color = "Unknown"

    if tracker is None:
        faces = detector(gray)
        if len(faces) > 0:
            face = faces[0]
            x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
            tracker = dlib.correlation_tracker()
            tracker.start_track(gray, dlib.rectangle(x1, y1, x2, y2))

    cv2.imshow("Original Video", frame)
    cv2.imshow("Grayscale", gray)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
