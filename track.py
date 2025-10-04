import cv2
from ultralytics import YOLO

# Load the YOLO11 model
model = YOLO("best.pt")
# Open the video file
# video_path = "./imgs/**.mp4"
video_path = "./imgs/**.avi"
cap = cv2.VideoCapture(video_path)


width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"Original video: {width}x{height}, FPS: {fps}")

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output_AVI_VIDEOS_1.avi', fourcc, fps, (width, height), isColor=True)

frame_count = 0

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        frame_count += 1
        print(f"Processing frame {frame_count}")

        # Run YOLO11 tracking on the frame, persisting tracks between frames
        results = model.track(frame, persist=True)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()


        print(f"Annotated frame size: {annotated_frame.shape[1]}x{annotated_frame.shape[0]}")


        if annotated_frame.shape[1] != width or annotated_frame.shape[0] != height:
            annotated_frame = cv2.resize(annotated_frame, (width, height))
            print("Resized frame to match original dimensions")


        out.write(annotated_frame)

    else:
        break

print(f"Processed {frame_count} frames")
cap.release()
out.release()
print("Video processing completed")