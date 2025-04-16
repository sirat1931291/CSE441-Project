# import cv2

# # Load the pre-trained Haar Cascade face detector
# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# # Start webcam (0 is the default webcam)
# cap = cv2.VideoCapture(0)

# while True:
#     # Read a frame from the webcam
#     ret, frame = cap.read()

#     # Convert the frame to grayscale (face detection works better in grayscale)
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     # Detect faces in the grayscale image
#     faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

#     # Draw rectangles around the faces
#     for (x, y, w, h) in faces:
#         cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

#     # Show the frame with detected faces
#     cv2.imshow('Face Detection', frame)

#     # Press 'q' to exit
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release the webcam and close the window
# cap.release()
# cv2.destroyAllWindows()


# import cv2

# # Load Haar Cascade face detector
# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# # Start the webcam
# cap = cv2.VideoCapture(0)

# while True:
#     # Read a frame from webcam
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Convert to grayscale
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     # Detect faces
#     faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

#     # Count faces
#     face_count = len(faces)

#     # Draw rectangles around faces
#     for (x, y, w, h) in faces:
#         cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

#     # Show count on the screen
#     cv2.putText(frame, f"Faces Detected: {face_count}", (10, 30),
#                 cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

#     # Show the frame
#     cv2.imshow("Face Counter", frame)

#     # Exit on pressing 'q'
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release camera and close window
# cap.release()
# cv2.destroyAllWindows()



# import cv2
# import numpy as np

# # Load Haar Cascade face detector
# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# # Start webcam
# cap = cv2.VideoCapture(0)

# # Minimum distance (in pixels) between two faces to be considered "close"
# DISTANCE_THRESHOLD = 100  # you can adjust this value

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     # Detect all faces
#     faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

#     # Convert face boxes to center points
#     centers = []
#     for (x, y, w, h) in faces:
#         center_x = x + w // 2
#         center_y = y + h // 2
#         centers.append((center_x, center_y))

#     # Check which faces are near each other
#     close_faces = []
#     for i, (cx1, cy1) in enumerate(centers):
#         for j, (cx2, cy2) in enumerate(centers):
#             if i != j:
#                 distance = np.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2)
#                 if distance < DISTANCE_THRESHOLD:
#                     close_faces.append(faces[i])
#                     break  # Only need one close neighbor to count it

#     # Draw only close faces
#     for (x, y, w, h) in close_faces:
#         cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

#     # Show count of close faces
#     cv2.putText(frame, f"Close Faces: {len(close_faces)}", (10, 30),
#                 cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

#     # Show the frame
#     cv2.imshow("Close Face Detection", frame)

#     # Exit on pressing 'q'
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()



import cv2
import numpy as np

# Load the Haar Cascade face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start the webcam
cap = cv2.VideoCapture(0)

# Distance threshold (adjust based on your camera setup)
DISTANCE_THRESHOLD = 100

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # Get face centers
    centers = [(x + w // 2, y + h // 2) for (x, y, w, h) in faces]

    # Store faces that are close to another face
    close_faces = []

    # If only 1 face is detected, still count it
    if len(faces) == 1:
        close_faces.append(faces[0])

    # If 2 or more faces, check for proximity
    elif len(faces) > 1:
        for i in range(len(centers)):
            for j in range(i + 1, len(centers)):
                x1, y1 = centers[i]
                x2, y2 = centers[j]
                distance = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
                if distance < DISTANCE_THRESHOLD:
                    # Add both faces if they're close
                    if faces[i] not in close_faces:
                        close_faces.append(faces[i])
                    if faces[j] not in close_faces:
                        close_faces.append(faces[j])

    # Draw rectangles around selected faces
    for (x, y, w, h) in close_faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

    # Show count on the screen
    cv2.putText(frame, f"Faces Counted: {len(close_faces)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    resized_frame = cv2.resize(frame, (480, 480))

    # Display the frame
    cv2.imshow("Face Detection - Near Faces Count", resized_frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()



# import cv2
# import numpy as np

# # Load Haar Cascade face detector
# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# # Start webcam
# cap = cv2.VideoCapture(0)

# # Set webcam resolution (optional)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# # Distance threshold (pixels)
# DISTANCE_THRESHOLD = 100

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Convert to grayscale
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     # Detect faces
#     detected_faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

#     # Convert faces to list of tuples
#     faces = [tuple(face) for face in detected_faces]

#     # Get center points of each face
#     centers = [(x + w // 2, y + h // 2) for (x, y, w, h) in faces]

#     close_faces = []

#     if len(faces) == 1:
#         close_faces.append(faces[0])

#     elif len(faces) > 1:
#         for i in range(len(centers)):
#             for j in range(i + 1, len(centers)):
#                 x1, y1 = centers[i]
#                 x2, y2 = centers[j]
#                 distance = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
#                 if distance < DISTANCE_THRESHOLD:
#                     if faces[i] not in close_faces:
#                         close_faces.append(faces[i])
#                     if faces[j] not in close_faces:
#                         close_faces.append(faces[j])

#     # Draw rectangles and count
#     for (x, y, w, h) in close_faces:
#         cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

#     # Show count
#     cv2.putText(frame, f"Faces Counted: {len(close_faces)}", (10, 30),
#                 cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

#     # Resize output display
#     resized_frame = cv2.resize(frame, (640, 480))
#     cv2.imshow("Face Detection - Near Faces Count", resized_frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release
# cap.release()
# cv2.destroyAllWindows()
