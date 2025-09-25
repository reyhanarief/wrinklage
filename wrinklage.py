import cv2

def resize_image(face, width, height):
    return cv2.resize(face, (width, height), interpolation=cv2.INTER_AREA)

def detect_landmarks(face):
    h, w, _ = face.shape
    x = w // 2
    y = h // 2
    landmarks = [
        ((x - 50, y - 7), (50, 15)), # Left eye
        ((x + 45, y - 7), (50, 15)), # Right eye
        ((x, y - 90), (120, 55)), # Forehead
        ((x - 60, y + 25), (35, 15)), # Left cheek
        ((x + 70, y + 25), (35, 15)) # Right cheek
    ]
    return landmarks

def apply_canny_on_landmarks(face, landmarks, thresholds):
    canny_results = []
    h, w = face.shape[:2]
    for ((center, (roi_width, roi_height)), (low_threshold, high_threshold)) in zip(landmarks, thresholds):
        (roi_x, roi_y) = center
        top_left = (max(roi_x - roi_width // 2, 0), max(roi_y - roi_height // 2, 0))
        bottom_right = (min(roi_x + roi_width // 2, w), min(roi_y + roi_height // 2, h))
        if top_left[0] >= bottom_right[0] or top_left[1] >= bottom_right[1]:
            continue
        roi = face[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
        if roi.size == 0 or roi.shape[0] < 2 or roi.shape[1] < 2:
            continue
        roi_edges = cv2.Canny(roi, int(low_threshold), int(high_threshold))
        canny_results.append((top_left, roi_edges))
    return canny_results

def categorize_age(wrinkle_percentage):
    if wrinkle_percentage > 15:
        return "Elderly"
    elif wrinkle_percentage > 8:
        return "Middle-aged"
    else:
        return "Young"

face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

if face_detector.empty():
    print("Failed to load face detector.")
    exit()

video_capture = cv2.VideoCapture(0)

if not video_capture.isOpened():
    print("Cannot open camera.")
    exit()

while True:
    ret, frame = video_capture.read()
    if not ret:
        print("Failed to grab frame from camera.")
        break

    grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(grayscale, 1.3, 5)

    for (x, y, w, h) in faces:
        detected_face = frame[y:y+h, x:x+w]
        expected_width = 250
        expected_height = 250
        resized_face = resize_image(detected_face, expected_width, expected_height)
        landmarks = detect_landmarks(resized_face)
        thresholds = [
            (10, 155),
            (10, 160),
            (8, 170),
            (6, 180),
            (6, 190)
        ]
        canny_landmarks = apply_canny_on_landmarks(resized_face, landmarks, thresholds)

        total_wrinkle_percentage = 0
        for (top_left, roi_edges) in canny_landmarks:
            roi_x, roi_y = top_left
            roi_width, roi_height = roi_edges.shape[1], roi_edges.shape[0]
            cv2.rectangle(resized_face, (roi_x, roi_y), (roi_x + roi_width, roi_y + roi_height), (0, 255, 0), 2)
            resized_face[roi_y:roi_y+roi_height, roi_x:roi_x+roi_width] = cv2.cvtColor(roi_edges, cv2.COLOR_GRAY2BGR)
            total_edge_pixel = cv2.countNonZero(roi_edges)
            total_area_pixel = roi_width * roi_height
            edge_percentage = (total_edge_pixel / total_area_pixel) * 100
            total_wrinkle_percentage += edge_percentage
        avg_wrinkle_percentage = total_wrinkle_percentage / len(canny_landmarks)
        age_category = categorize_age(avg_wrinkle_percentage)
        cv2.putText(frame, f"Age Category: {age_category}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 10, 10), 2)

    cv2.imshow("Age Categorization Based on Facial Wrinkles", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

video_capture.release()
cv2.destroyAllWindows()