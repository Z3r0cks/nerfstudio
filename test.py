import cv2
import os

# Video-Datei laden
video_path = 'C:/Users/free3D/Desktop/Patrick_Kaserer/Masterthesis/position_test/1m_chair_video/video.mp4'  # Pfad zu deinem Video
output_folder = 'C:/Users/free3D/Desktop/Patrick_Kaserer/Masterthesis/position_test/1m_chair_video/frames'  # Ordner, in den die Frames gespeichert werden

# Erstelle den Ordner, wenn er nicht existiert
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Video mit OpenCV öffnen
cap = cv2.VideoCapture(video_path)

frame_number = 0

# Video-Frames durchlaufen
while True:
    ret, frame = cap.read()  # Frame lesen
    if not ret:
        break  # Beende, wenn das Video zu Ende ist

    # Speichere jeden Frame als Bild
    frame_path = os.path.join(output_folder, f'frame_{frame_number:04d}.jpg')
    cv2.imwrite(frame_path, frame)

    frame_number += 1

# Freigeben des Videos
cap.release()

print(f'Extrahierte {frame_number} Frames und in {output_folder} gespeichert.')
