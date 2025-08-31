import os
import cv2
import random
import base64
from ultralytics import YOLO
from tqdm import tqdm
from IPython.display import HTML, display
import argparse

def create_and_play_video(model_path, image_folder, output_name, play_in_colab=False):
    """
    Processes a folder of images, runs YOLO inference, and creates/plays a video.
    """
    print("Starting video creation process...")

    # --- Load Your Trained Model ---
    try:
        model = YOLO(model_path)
        print("Successfully loaded the trained model.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # --- Get and Sort Image Files ---
    try:
        image_files = sorted([f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.png'))])
        if not image_files:
            print(f"Error: No images found in {image_folder}")
            return
        print(f"Found {len(image_files)} images to process.")
    except FileNotFoundError:
        print(f"Error: The directory {image_folder} was not found.")
        return
        
    # --- Set up Video Writer ---
    first_image_path = os.path.join(image_folder, image_files[0])
    frame = cv2.imread(first_image_path)
    if frame is None:
        print(f"Error: Could not read the first image at {first_image_path}")
        return
        
    height, width, _ = frame.shape
    fps = 15

    fourcc = cv2.VideoWriter_fourcc(*'avc1') 
    video_writer = cv2.VideoWriter(output_name, fourcc, fps, (width, height))
    print(f"Video writer initialized for {output_name} with {fps} FPS.")

    # --- Process Each Frame and Write to Video ---
    for image_file in tqdm(image_files, desc="Processing frames"):
        image_path = os.path.join(image_folder, image_file)
        results = model(image_path, verbose=False)
        annotated_frame = results[0].plot()
        video_writer.write(annotated_frame)

    video_writer.release()
    print("\n----------------------------------")
    print(f"✅ Success! Video created: {output_name}")

    # --- Embed and Play the Video if in a compatible environment ---
    if play_in_colab:
        print("\nEmbedding video for playback...")
        try:
            if os.path.exists(output_name) and os.path.getsize(output_name) > 0:
                with open(output_name, "rb") as video_file:
                    video_data = video_file.read()
                    b64_video = base64.b64encode(video_data).decode()
                video_html = f"""
                <video width=600 controls autoplay loop>
                    <source src="data:video/mp4;base64,{b64_video}" type="video/mp4">
                    Your browser does not support the video tag.
                </video>
                """
                display(HTML(video_html))
                print("Playback should start above.")
            else:
                print(f"Video file '{output_name}' was not created or is empty.")
        except Exception as e:
            print(f"An error occurred during playback: {e}")

if __name__ == '__main__':
    # This block allows running the script from the command line
    parser = argparse.ArgumentParser(description="Create a video from YOLOv10 inference results.")
    parser.add_argument('--model', type=str, default='runs/detect/train/weights/best.pt', help='Path to your trained .pt model file.')
    parser.add_argument('--images', type=str, default='data/images/val', help='Path to the folder of validation images.')
    parser.add_argument('--output', type=str, default='traffic_detection_output.mp4', help='Name of the output video file.')
    
    # In a Colab notebook, you wouldn't use argparse. Instead, you would set the paths directly.
    # For local execution, you can run:
    # python video_inference.py --model path/to/best.pt --images path/to/images --output my_video.mp4
    
    # Example of how to run it in a script/notebook without command line args:
    # Set this to True if you are in a Colab/Jupyter environment
    IS_COLAB_NOTEBOOK = 'google.colab' in str(get_ipython())

    if IS_COLAB_NOTEBOOK:
        # Hardcode paths for Colab execution
        model_path = 'runs/detect/train/weights/best.pt'
        image_folder = '/content/yolo_dataset/images/val/'
        output_name = 'drone_traffic_detection.mp4'
        create_and_play_video(model_path, image_folder, output_name, play_in_colab=True)
    else:
        args = parser.parse_args()
        create_and_play_video(args.model, args.images, args.output)
