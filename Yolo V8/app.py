import gradio as gr
import cv2
import torch
import numpy as np
from ultralytics import YOLO
import random
import tempfile
import time
from tqdm import tqdm

# Downloading the images and video from Google Drive
def download_file(url, save_name):
    import gdown
    gdown.download(url, save_name, quiet=False)

file_urls = [
    ('https://drive.google.com/uc?id=1nqF_ORRO3bHfjccK-HSwx6NsLnIz7r4z', 'image_0.jpg'),
    ('https://drive.google.com/uc?id=1dIYLlJ42dU_7yC8YdqrmIBiEHCBxN3rM', 'image_1.jpg'),
    ('https://drive.google.com/uc?id=1BW9y9l09BtNdbLI-irH9ksVE6FUBXDXn', 'video.mp4')
]

for url, save_name in file_urls:
    download_file(url, save_name)

# Initialize the model
model = YOLO('best.pt')

# Define a color for each class
class_colors = {}
for class_id in range(len(model.names)):
    class_colors[class_id] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

def draw_boxes(image, boxes, labels, colors):
    for box, label, color in zip(boxes, labels, colors):
        top_left = (int(box[0]), int(box[1]))
        bottom_right = (int(box[2]), int(box[3]))
        cv2.rectangle(image, top_left, bottom_right, color, 2)
        cv2.putText(image, label, (top_left[0], top_left[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return image

def show_preds_image(image_path, conf_threshold=0.25, iou_threshold=0.45):
    image = cv2.imread(image_path)
    results = model(image, conf=conf_threshold, iou=iou_threshold)
    boxes = []
    labels = []
    colors = []
    num_boxes_detected = 0
    for result in results:
        for box, cls, conf in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.cls.cpu().numpy(), result.boxes.conf.cpu().numpy()):
            boxes.append(box)
            label = result.names[int(cls)]
            labels.append(f'{label} {conf:.2f}')
            colors.append(class_colors[int(cls)])
            num_boxes_detected += 1

    image = draw_boxes(image, boxes, labels, colors)

    # Display the count on the top right corner
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 2
    font_color = (255, 255, 255)
    count_text = f'Boxes Detected: {num_boxes_detected}'
    text_size = cv2.getTextSize(count_text, font, font_scale, font_thickness)[0]
    text_x = image.shape[1] - text_size[0] - 10
    text_y = text_size[1] + 10
    cv2.putText(image, count_text, (text_x, text_y), font, font_scale, font_color, font_thickness, lineType=cv2.LINE_AA)

    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def show_preds_video(video_path, conf_threshold=0.25, iou_threshold=0.45):
    start_time = time.time()
    print("Starting video processing...")

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
        output_path = tmp_file.name

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    batch_size = 4  # Adjust this based on your GPU memory
    frames = []

    for _ in tqdm(range(total_frames), desc="Processing frames"):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

        if len(frames) == batch_size or not ret:
            batch_results = model(frames, conf=conf_threshold, iou=iou_threshold)

            for frame, result in zip(frames, batch_results):
                frame_copy = frame.copy()
                boxes = []
                labels = []
                colors = []
                num_boxes_detected = 0

                for box, cls, conf in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.cls.cpu().numpy(), result.boxes.conf.cpu().numpy()):
                    boxes.append(box)
                    label = result.names[int(cls)]
                    labels.append(f'{label} {conf:.2f}')
                    colors.append(class_colors[int(cls)])
                    num_boxes_detected += 1

                frame_copy = draw_boxes(frame_copy, boxes, labels, colors)

                # Display the count on the top right corner
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1
                font_thickness = 2
                font_color = (255, 255, 255)
                count_text = f'Boxes Detected: {num_boxes_detected}'
                text_size = cv2.getTextSize(count_text, font, font_scale, font_thickness)[0]
                text_x = frame_copy.shape[1] - text_size[0] - 10
                text_y = text_size[1] + 10
                cv2.putText(frame_copy, count_text, (text_x, text_y), font, font_scale, font_color, font_thickness, lineType=cv2.LINE_AA)

                out.write(frame_copy)

            frames = []

    cap.release()
    out.release()

    end_time = time.time()
    total_time = end_time - start_time
    print(f"Video processing completed in {total_time:.2f} seconds.")

    return output_path

inputs_image = [
    gr.components.Image(type="filepath", label="Input Image"),
    gr.components.Slider(minimum=0.0, maximum=1.0, value=0.25, step=0.05, label="Confidence Threshold"),
    gr.components.Slider(minimum=0.0, maximum=1.0, value=0.45, step=0.05, label="IOU Threshold")
]
outputs_image = [
    gr.components.Image(type="numpy", label="Output Image"),
]
interface_image = gr.Interface(
    fn=show_preds_image,
    inputs=inputs_image,
    outputs=outputs_image,
    title="Traffic Detection - Image",
    examples=[['image_0.jpg'], ['image_1.jpg']],
)

inputs_video = [
    gr.Video(label="Input Video"),
    gr.Slider(minimum=0.0, maximum=1.0, value=0.25, step=0.05, label="Confidence Threshold"),
    gr.Slider(minimum=0.0, maximum=1.0, value=0.45, step=0.05, label="IOU Threshold")
]
outputs_video = gr.Video(label="Output Video")

interface_video = gr.Interface(
    fn=show_preds_video,
    inputs=inputs_video,
    outputs=outputs_video,
    title="Traffic Detection - Video",
    examples=[['video.mp4']],
)

demo = gr.TabbedInterface([interface_image, interface_video], ["Image", "Video"])
demo.launch(share=True)