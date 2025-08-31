YOLOv10 Real-Time Traffic Detection
This project showcases a fine-tuned YOLOv10 model for real-time traffic object detection. The model was trained on the challenging Multi-View Traffic Intersection Dataset (MTID) and is capable of identifying cars, buses, lorries, and bicycles with high accuracy.

This repository serves as a demonstration of a complete MLOps pipeline, from data preprocessing and cleaning to model training, validation, and inference.

🚀 Features
State-of-the-Art Model: Utilizes YOLOv10n, a powerful and efficient object detection model.

Custom-Trained: Fine-tuned on a large-scale, real-world traffic dataset.

High Performance: Achieved 79.2% mAP50-95 on the validation set.

Inference Ready: Includes a script to run the model on a folder of images and generate a video of the results.

🛠️ Setup & Installation
Follow these steps to set up the project locally.

1. Clone the Repository

git clone [https://github.com/YOUR_USERNAME/YOLOv10-Traffic-Detection.git](https://github.com/YOUR_USERNAME/YOLOv10-Traffic-Detection.git)
cd YOLOv10-Traffic-Detection

2. Create a Virtual Environment
It's highly recommended to use a virtual environment to manage dependencies.

python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

3. Install Dependencies
This project uses a requirements.txt file to manage all necessary Python packages.

pip install -r requirements.txt

4. Download the Trained Model
You will need the trained model weights file (best.pt). Download it from your project's runs folder in Google Colab and place it in the root directory of this repository.

🏃‍♂️ How to Run
The video_inference.py script is used to run the model on a directory of images and generate a video.

1. Prepare Your Data

Create a folder (e.g., data/my_images/)

Place the images or video frames you want to process inside this folder.

2. Run the Inference Script
Execute the script from your terminal, pointing it to your model file and image folder.

python video_inference.py --model best.pt --images data/my_images/ --output my_results.mp4

The script will process each image and save a new video file named my_results.mp4 in the root directory.

📊 Model Performance
The model was trained for 50 epochs on a Google Colab T4 GPU, achieving the following metrics on the validation set:

Class

mAP50-95

Precision

Recall

All

0.792

0.889

0.877

Car

0.875

0.912

0.905

Bus

0.941

0.926

0.972

Lorry

0.881

0.858

0.903

Bicycle

0.472

0.859

0.727

🙏 Acknowledgements and Dataset
This project would not have been possible without the high-quality Multi-View Traffic Intersection Dataset (MTID). All credit for the data collection and initial annotation goes to the original authors.

Dataset Homepage: MTID - Multi-View Traffic Intersection Dataset

Citation: Jensen, M. B., Møgelmose, A., & Moeslund, T. B. (2019). Presenting the Multi-View Traffic Intersection Dataset (MTID). In 2019 IEEE International Conference on Image Processing (ICIP).

This project is a demonstration of skills in computer vision and deep learning, created as part of my M.Tech in CSE at VIT Vellore.
