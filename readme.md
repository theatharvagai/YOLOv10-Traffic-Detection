# 🚀 YOLOv10 Real-Time Traffic Detection

🚦 **Real-Time Object Detection** • 🤖 **YOLOv10n** • 📊 **MTID Dataset** • 📦 **MLOps Pipeline**

---

This project showcases a **fine-tuned YOLOv10 model** for real-time traffic object detection. The model identifies **cars, buses, lorries, and bicycles** with high accuracy, trained on the challenging Multi-View Traffic Intersection Dataset (MTID).

This repository demonstrates a complete MLOps pipeline, from data preprocessing to model training, validation, and inference.

<br>
_Add a GIF or image here showing your model in action!_

---

## 📌 Key Features

* **✅ State-of-the-Art Model:** Utilizes **YOLOv10n**, a powerful and efficient object detection model.
* **✅ Custom-Trained:** Fine-tuned on a large-scale, real-world traffic dataset for superior performance.
* **✅ High Performance:** Achieved an impressive **79.2% mAP50-95** on the validation set.
* **✅ Inference Ready:** Comes with a simple Python script to run the model on your own images and generate a detection video.

---

## 🛠️ Technologies & Frameworks Used

* **💻 Python**
* **🤖 PyTorch**
* **🚀 YOLOv10**
* ** OpenCV** for video processing
* **📊 NumPy** for numerical operations

---

## 📊 Model Performance

The model was trained for **50 epochs** on a Google Colab **T4 GPU**.

| Class     | mAP50-95  | Precision | Recall  |
| :-------- | :-------: | :-------: | :-----: |
| 🌍 **All** | **0.792** | **0.889** | **0.877** |
| 🚗 Car   | 0.875     | 0.912     | 0.905   |
| 🚌 Bus   | 0.941     | 0.926     | 0.972   |
| 🚚 Lorry | 0.881     | 0.858     | 0.903   |
| 🚲 Bicycle | 0.472     | 0.859     | 0.727   |

---

## ⚙️ Setup & Installation

Follow these steps to get the project running on your local machine.

1.  **Clone the Repository**
    ```sh
    git clone [https://github.com/YOUR_USERNAME/YOLOv10-Traffic-Detection.git](https://github.com/YOUR_USERNAME/YOLOv10-Traffic-Detection.git)
    cd YOLOv10-Traffic-Detection
    ```

2.  **Create and Activate a Virtual Environment**
    ```sh
    # Create the virtual environment
    python -m venv venv

    # Activate it
    # On macOS/Linux:
    source venv/bin/activate
    # On Windows:
    .\venv\Scripts\activate
    ```

3.  **Install Dependencies**
    ```sh
    pip install -r requirements.txt
    ```

4.  **Download the Trained Model**
    Download the trained weights file (`best.pt`) from your project's `runs` folder in Google Colab and place it in the root directory of this repository.

---

## 🚀 How to Run Inference

1.  **Prepare Your Data**
    * Create a folder for your images (e.g., `data/my_images/`).
    * Place all the images or video frames you want to process inside this folder.

2.  **Run the Inference Script**
    * Execute the script from your terminal. Point it to your model file, image folder, and desired output video name.
    ```sh
    python video_inference.py --model best.pt --images data/my_images/ --output my_results.mp4
    ```
    * The script will process each image and save a new video file named `my_results.mp4` in the root directory.

---

## 🙏 Acknowledgements & Dataset

This project was made possible by the high-quality **Multi-View Traffic Intersection Dataset (MTID)**. All credit for data collection and annotation goes to the original authors.

* **Dataset Homepage:** [MTID - Multi-View Traffic Intersection Dataset](https://vap.aau.dk/mtid/)
* **Citation:** Jensen, M. B., Møgelmose, A., & Moeslund, T. B. (2019). Presenting the Multi-View Traffic Intersection Dataset (MTID). In *2019 IEEE International Conference on Image Processing (ICIP).*

---

## 👨‍💻 About This Project

This project is a demonstration of skills in computer vision and deep learning, created as part of my **M.Tech in CSE at VIT Vellore**.
