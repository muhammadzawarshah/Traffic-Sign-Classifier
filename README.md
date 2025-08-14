
🚦 Traffic Sign Classifier (HOG + SVM)
======================================

A Streamlit-based web application that classifies traffic sign images using Histogram of Oriented Gradients (HOG) 
for feature extraction and Support Vector Machine (SVM) for classification. The model is trained offline, and the 
web app provides real-time predictions with a visual explanation of the HOG features.

--------------------------------------
📌 Features
--------------------------------------
- Upload any traffic sign image (.jpg, .jpeg, .png, .webp)
- Real-time prediction of traffic sign class
- HOG visualization to understand feature extraction
- Streamlit-powered UI for an interactive experience
- Uses OpenCV + scikit-image for preprocessing and visualization

--------------------------------------
🛠️ Tech Stack
--------------------------------------
- Python
- OpenCV – Image loading, preprocessing, and HOG computation
- scikit-image – HOG visualization
- scikit-learn – Model training (SVM)
- Streamlit – Web application interface
- joblib – Model serialization

--------------------------------------
📂 Project Structure
--------------------------------------
traffic-sign-classifier/
 ┣ app.py                      # Streamlit app
 ┣ traffic_sign_model_with_params.pkl  # Trained SVM model + HOG params
 ┣ requirements.txt            # Python dependencies
 ┣ .gitignore                  # Ignored files/folders
 ┗ README.txt                  # Project documentation

--------------------------------------
🚀 Installation & Usage
--------------------------------------
1️⃣ Clone the repository:
    git clone https://github.com/muhammadzawarshah/Traffic-Sign-Classifier.git
    cd traffic-sign-classifier

2️⃣ Install dependencies:
    pip install -r requirements.txt

3️⃣ Run the Streamlit app:
    streamlit run app.py

4️⃣ Upload an image and get predictions!

--------------------------------------
📊 How It Works
--------------------------------------
1. HOG Feature Extraction
   - Converts the image to grayscale
   - Extracts edge orientation histograms
   - Flattens into a feature vector for the SVM

2. SVM Prediction
   - Pre-trained SVM model classifies the traffic sign based on HOG features

3. Visualization
   - Displays original image and HOG feature map side-by-side

--------------------------------------
📌 Dependencies
--------------------------------------
opencv-python
streamlit
numpy
scikit-image
scikit-learn
joblib

--------------------------------------
📜 License
--------------------------------------
This project is licensed under the MIT License – feel free to use, modify, and share.

--------------------------------------
🤝 Contributing
--------------------------------------
Pull requests are welcome. For major changes, please open an issue first to discuss your ideas.
