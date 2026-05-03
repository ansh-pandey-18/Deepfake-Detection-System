# Deepfake Detection System for Media Authenticity

## Project Overview
The **Deepfake Detection System for Media Authenticity** is an advanced cyber-forensics application designed to identify AI-generated or manipulated media. Built with a professional, interactive dashboard, the system empowers users, researchers, and authorities to confidently analyze both images and videos. By utilizing deep learning models combined with explainable AI (Grad-CAM), the application not only delivers a verdict but also visualizes *why* a specific piece of media was flagged as manipulated.

## Team Members
* Ansh
* Aryan
* Kanishk

## Features
* **Image Detection:** Predicts whether an image is Real or AI-Generated using specialized neural networks (CNN, EfficientNet, and EfficientNet Art) tailored for different input categories (e.g., General Photos, AI Faces, Artwork).
* **Video Detection (Frame-by-Frame):** Intelligently extracts frames from videos and analyzes them sequentially. Identifies exactly which segments of a video contain AI-manipulated content.
* **Explainable AI (Grad-CAM):** Generates heatmaps indicating the specific regions of an image that influenced the model's prediction, providing transparency in the detection process.
* **Timeline Detection:** For videos, generates an interactive timeline pinpointing the exact timestamps where deepfakes were detected.
* **Confidence Stability Metrics:** Calculates prediction spread across passes or standard deviations across frames to determine the reliability (🟢 Very Stable, 🟡 Moderately Stable, 🔴 Uncertain) of the detection.
* **Evidence Report Generation:** Generates and downloads a comprehensive, professional PDF forensic report containing detection metadata, risk assessments, Grad-CAM summaries, and recommended actions.
* **Authority Reporting Panels:** Direct, secure redirection links to official Cybercrime portals and Platform reporting pages (Meta, etc.) for immediate action on malicious content.

## Technologies Used
* **Frontend/Dashboard:** Streamlit, Plotly, CSS
* **Machine Learning & AI:** TensorFlow, Keras, OpenCV
* **Data Processing:** NumPy, Pillow (PIL)
* **Report Generation:** fpdf2

## Methodology
The system utilizes a multi-model architecture. An initial categorization (General Photo, AI Face, Artwork) directs the input to the most appropriate, pre-trained neural network:
1. **CNN:** Optimized for identifying manipulations in standard photography and faces.
2. **EfficientNet:** Tuned for high-resolution AI face generation detection.
3. **EfficientNet Art:** Specialized in distinguishing human-drawn art from AI-generated imagery.

For explainability, Gradient-weighted Class Activation Mapping (Grad-CAM) is dynamically applied to the final convolutional layers of the selected model. For videos, the system extracts key frames, applies the detection pipeline to each, and aggregates the confidence scores to determine an overall verdict and timeline.

## How to Run
1. Clone the repository to your local machine.
2. Navigate to the project directory:
   ```bash
   cd "Deepfake Detection System"
   ```
3. Create and activate a Python virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```
4. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
5. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```

## Future Scope
* **Audio Deepfake Detection:** Expanding the system to analyze and detect synthesized or cloned voices in video files.
* **Real-time Stream Analysis:** Implementing optimized inference pipelines for live video feeds and browser extensions.
* **Enhanced Model Ensembles:** Integrating Vision Transformers (ViTs) alongside the existing CNN/EfficientNet models for increased accuracy against next-generation diffusion models.
