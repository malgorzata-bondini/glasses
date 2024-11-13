import cv2
import numpy as np
from PIL import Image
import os

def overlay_image_alpha(img, img_overlay, pos):
    """Overlay img_overlay on top of img at the position specified by pos with transparency."""
    x, y = pos
    y1, y2 = max(0, y), min(img.shape[0], y + img_overlay.shape[0])
    x1, x2 = max(0, x), min(img.shape[1], x + img_overlay.shape[1])

    overlay_image = img_overlay[y1-y:y2-y, x1-x:x2-x]
    img_crop = img[y1:y2, x1:x2]

    alpha = overlay_image[:, :, 3] / 255.0
    alpha_inv = 1.0 - alpha

    for c in range(0, 3):
        img_crop[:, :, c] = (alpha * overlay_image[:, :, c] +
                             alpha_inv * img_crop[:, :, c])

    img[y1:y2, x1:x2] = img_crop
    return img

def load_glasses_image(filename):
    """Load an image file and convert it to a format suitable for overlay."""
    img = Image.open(filename).convert("RGBA")
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGBA2BGRA)

def rank_glasses_for_oval_face(glasses_width, face_width, face_height, glasses_index):
    """Rank glasses for an oval face shape based on dimensions and proportions."""
    ideal_ratio = 1.5
    face_ratio = face_width / face_height
    glasses_face_ratio = glasses_width / face_width
    width_score = 10 - abs(glasses_face_ratio - 1) * 10
    ratio_score = 10 - abs(face_ratio - ideal_ratio) * 5
    unique_factor = [0.8, 1.0, 1.2, 0.9][glasses_index]
    total_score = (width_score * 0.5 + ratio_score * 0.3 + unique_factor * 2)
    return min(max(total_score, 0), 10)  # Ensure score is between 0 and 10

def create_sidebar(glasses_images, current_glasses, frame_height):
    sidebar = np.zeros((frame_height, 100, 3), dtype=np.uint8)
    thumbnail_height = frame_height // len(glasses_images)
    for i, img in enumerate(glasses_images):
        y = i * thumbnail_height
        resized = cv2.resize(img[:, :, :3], (80, thumbnail_height - 20))
        sidebar[y+10:y+thumbnail_height-10, 10:90] = resized
        if i == current_glasses:
            cv2.rectangle(sidebar, (5, y+5), (95, y+thumbnail_height-5), (0, 255, 0), 2)
    return sidebar

def main():
    glasses_base_dir = r"C:\Users\plmala\OneDrive - Coloplast A S\Desktop\Python\Glasses_ComputerVision"
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)
    glasses_images = [
        load_glasses_image(os.path.join(glasses_base_dir, "glasses1.png")),
        load_glasses_image(os.path.join(glasses_base_dir, "glasses2.png")),
        load_glasses_image(os.path.join(glasses_base_dir, "glasses3.png")),
        load_glasses_image(os.path.join(glasses_base_dir, "glasses4.png"))
    ]
    current_glasses = 0 
    
    def mouse_callback(event, x, y, flags, param):
        nonlocal current_glasses
        if event == cv2.EVENT_LBUTTONDOWN:
            if x > frame.shape[1]: 
                clicked_glasses = y // (frame.shape[0] // len(glasses_images))
                if clicked_glasses < len(glasses_images):
                    current_glasses = clicked_glasses

    cv2.namedWindow('Glasses Try-On App')
    cv2.setMouseCallback('Glasses Try-On App', mouse_callback)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        sidebar = create_sidebar(glasses_images, current_glasses, frame.shape[0])

        for (x, y, w, h) in faces:
            glasses = cv2.resize(glasses_images[current_glasses], (w, int(h / 1.8)))
            glasses_pos = (x, y + int(h / 10))
            frame = overlay_image_alpha(frame, glasses, glasses_pos)
            score = rank_glasses_for_oval_face(glasses.shape[1], w, h, current_glasses)
    
            cv2.rectangle(frame, (10, 10), (250, 80), (0, 0, 0), -1)
            cv2.rectangle(frame, (10, 10), (250, 80), (0, 255, 0), 2)
            
            cv2.putText(frame, f"Score:", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame, f"{score:.1f}/10", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        combined_frame = np.hstack((frame, sidebar))
        cv2.imshow('Glasses Try-On App', combined_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('n'):
            current_glasses = (current_glasses + 1) % len(glasses_images)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
