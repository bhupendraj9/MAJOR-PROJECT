import cv2
import numpy as np
import time
from ultralytics import YOLO
from tensorflow.keras.models import load_model
import pyttsx3

# --- Voice Output Setup ---
engine = pyttsx3.init()
def speak(text):
    engine.say(text)
    engine.runAndWait()

# --- Currency Detection Setup ---
CURRENCY_CLASS_LABELS = ['10', '100', '20', '200', '2000', '50', '500']
CURRENCY_IMG_SIZE = (224, 224)

def load_currency_detector(model_path):
    model = load_model(model_path)
    def detector(frame):
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, CURRENCY_IMG_SIZE)
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)
        preds = model.predict(img)
        class_idx = np.argmax(preds)
        confidence = float(np.max(preds))
        label = CURRENCY_CLASS_LABELS[class_idx] if class_idx < len(CURRENCY_CLASS_LABELS) else 'Unknown'
        return label, confidence
    return detector

# --- Camera Selection (with iVCam support) ---
def select_camera():
    print("Select camera:")
    print("1. Laptop webcam")
    print("2. Mobile camera (iVCam)")
    choice = input("Enter choice (1 or 2): ").strip()
    if choice == "2":
        print("\niVCam (Mobile Camera) Setup:")
        print("1. Install and open the iVCam app on your phone.")
        print("2. Install and open the iVCam Windows client on your PC.")
        print("3. Make sure both devices are on the same WiFi network.")
        print("4. Wait for the connection. Your phone's camera feed should appear in the iVCam Windows client.\n")
        print("Searching for iVCam virtual webcam...")
        cap = None
        camera_indices = []
        for idx in range(3):
            temp_cap = cv2.VideoCapture(idx)
            ret, frame = temp_cap.read()
            if ret and frame is not None:
                camera_indices.append(idx)
            temp_cap.release()
        if not camera_indices:
            print("ERROR: Could not find iVCam webcam. Make sure iVCam is running and connected.")
            exit(1)
        elif len(camera_indices) == 1:
            print(f"iVCam found at webcam index {camera_indices[0]}.")
            cap = cv2.VideoCapture(camera_indices[0])
        else:
            print("Multiple camera devices found:")
            for i, idx in enumerate(camera_indices):
                print(f"{i+1}: Camera index {idx}")
            selected = input(f"Select camera to use for iVCam (1-{len(camera_indices)}): ").strip()
            try:
                selected_idx = camera_indices[int(selected)-1]
            except Exception:
                print("Invalid selection. Defaulting to first available camera.")
                selected_idx = camera_indices[0]
            print(f"Using camera index {selected_idx} for iVCam.")
            cap = cv2.VideoCapture(selected_idx)
    else:
        print("Using laptop webcam...")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("ERROR: Could not open laptop webcam.")
            exit(1)
    return cap

# --- Main Integration ---
def main():
    print("Combined Detection System")
    print("-------------------------")
    print("Select detection mode:")
    print("1. Road Sign Detection (YOLOv8, best.pt)")
    print("2. Object Detection (YOLOv8x)")
    print("3. Currency Detection (Custom CNN)")
    print("4. All Modes (Parallel)")
    mode = input("Enter choice (1-4): ").strip()

    # Model paths
    road_sign_model_path = "SIGN-D/best.pt"
    object_model_path = "OBJ&CUR-D/yolov8x.pt"
    currency_model_path = "OBJ&CUR-D/custom_cnn_model.h5"

    # Load models as needed
    if mode == "1":
        print("Loading road sign model...")
        road_sign_model = YOLO(road_sign_model_path)
    elif mode == "2":
        print("Loading object detection model...")
        object_model = YOLO(object_model_path)
    elif mode == "3":
        print("Loading currency detection model...")
        currency_detector = load_currency_detector(currency_model_path)
    elif mode == "4":
        print("Loading all models...")
        road_sign_model = YOLO(road_sign_model_path)
        object_model = YOLO(object_model_path)
        currency_detector = load_currency_detector(currency_model_path)
    else:
        print("Invalid choice.")
        return

    cap = select_camera()
    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("Error reading frame. Check camera connection.")
            time.sleep(1)
            continue

        display_frame = frame.copy()
        output_texts = []

        if mode == "1" or mode == "4":
            # Road sign detection
            results = road_sign_model.predict(frame, imgsz=640, conf=0.3)
            for result in results:
                boxes = result.boxes.xyxy.cpu().numpy() if hasattr(result.boxes, 'xyxy') else []
                classes = result.boxes.cls.cpu().numpy() if hasattr(result.boxes, 'cls') else []
                confs = result.boxes.conf.cpu().numpy() if hasattr(result.boxes, 'conf') else []
                for box, cls, conf in zip(boxes, classes, confs):
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    label = f"Sign: {road_sign_model.names[int(cls)]} ({conf:.2f})"
                    cv2.putText(display_frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                    output_texts.append(label)
                    # Speak only the class name, replace underscores with spaces
                    sign_name = str(road_sign_model.names[int(cls)]).replace('_', ' ')
                    speak(sign_name)
        if mode == "2" or mode == "4":
            # Object detection
            results = object_model.predict(frame, imgsz=640, conf=0.3)
            for result in results:
                boxes = result.boxes.xyxy.cpu().numpy() if hasattr(result.boxes, 'xyxy') else []
                classes = result.boxes.cls.cpu().numpy() if hasattr(result.boxes, 'cls') else []
                confs = result.boxes.conf.cpu().numpy() if hasattr(result.boxes, 'conf') else []
                for box, cls, conf in zip(boxes, classes, confs):
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    label = f"Obj: {object_model.names[int(cls)]} ({conf:.2f})"
                    cv2.putText(display_frame, label, (x1, y2+20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)
                    output_texts.append(label)
                    # Speak only the object class name
                    speak(str(object_model.names[int(cls)]))
        if mode == "3" or mode == "4":
            # Currency detection
            label, confidence = currency_detector(frame)
            output_text = f"Currency: {label} (Conf: {confidence:.2f})"
            cv2.putText(display_frame, output_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            output_texts.append(output_text)
            # Speak only the currency label
            speak(str(label))

        cv2.imshow('Combined Detection', display_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
