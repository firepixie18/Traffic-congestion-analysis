import gradio as gr
import cv2
import requests
import os
from ultralytics import YOLO

file_urls = [
    'https://drive.google.com/uc?export=download&id=1nqF_ORRO3bHfjccK-HSwx6NsLnIz7r4z',
    'https://drive.google.com/uc?export=download&id=1dIYLlJ42dU_7yC8YdqrmIBiEHCBxN3rM',
    'https://drive.google.com/uc?export=download&id=1BW9y9l09BtNdbLI-irH9ksVE6FUBXDXn'
]

def download_file(url, save_name):
    if not os.path.exists(save_name):
        file = requests.get(url)
        open(save_name, 'wb').write(file.content)

for i, url in enumerate(file_urls):
    if 'mp4' in file_urls[i]:
        download_file(file_urls[i], 'video.mp4')
    else:
        download_file(file_urls[i], f'image_{i}.jpg')

# Load your trained model
model = YOLO('best.pt')
path = [['image_0.jpg'], ['image_1.jpg']]
video_path = [['video.mp4']]

def show_preds_image(image_path):
    image = cv2.imread(image_path)
    outputs = model.predict(source=image_path)
    results = outputs[0].cpu().numpy()
    for i, det in enumerate(results.boxes.xyxy):
        cv2.rectangle(
            image,
            (int(det[0]), int(det[1])),
            (int(det[2]), int(det[3])),
            color=(0, 0, 255),
            thickness=2,
            lineType=cv2.LINE_AA
        )
        cv2.putText(
            image,
            f"{results.names[int(det[-1])]}: {det[-2]:.2f}",
            (int(det[0]), int(det[1]) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

inputs_image = [
    gr.components.Image(type="filepath", label="Input Image"),
]
outputs_image = [
    gr.components.Image(type="numpy", label="Output Image"),
]
interface_image = gr.Interface(
    fn=show_preds_image,
    inputs=inputs_image,
    outputs=outputs_image,
    title="Multi-class Detector",
    examples=path,
    cache_examples=False,
)

def show_preds_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_copy = frame.copy()
        outputs = model.predict(source=frame)
        results = outputs[0].cpu().numpy()
        for i, det in enumerate(results.boxes.xyxy):
            cv2.rectangle(
                frame_copy,
                (int(det[0]), int(det[1])),
                (int(det[2]), int(det[3])),
                color=(0, 0, 255),
                thickness=2,
                lineType=cv2.LINE_AA
            )
            cv2.putText(
                frame_copy,
                f"{results.names[int(det[-1])]}: {det[-2]:.2f}",
                (int(det[0]), int(det[1]) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )
        frames.append(cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames

inputs_video = [
    gr.components.Video(type="filepath", label="Input Video"),
]
outputs_video = [
    gr.components.Video(type="numpy", label="Output Video"),
]
interface_video = gr.Interface(
    fn=show_preds_video,
    inputs=inputs_video,
    outputs=outputs_video,
    title="Multi-class Detector",
    examples=video_path,
    cache_examples=False,
)

gr.TabbedInterface(
    [interface_image, interface_video],
    tab_names=['Image Inference', 'Video Inference']
).queue().launch()