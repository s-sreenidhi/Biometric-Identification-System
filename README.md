# Biometric-Identification-System





## Tech Stack 🛠️

* **Frontend:** HTML, CSS, Bootstrap 5
* **Backend:** Flask (Python) -  *Backend specifics coming soon!  We're polishing the engine room.*
* **Biometric Models:** ResNet50, ViT, Custom CNN (Wavelet-based features) - *Model details are under wraps for now...  🤫  Think magic.*


## Features 🚀

* **Face Detection:**  Combines ResNet50 and ViT for high-accuracy face recognition, even with challenging lighting and expressions. 😎
* **Iris Detection:** Employs wavelet-based texture analysis and ViT for robust and spoof-resistant iris recognition.  👁️‍🗨️
* **Fingerprint Detection:** Leverages wavelet-transformed features and a custom CNN for fast and reliable fingerprint matching. 🫗
* **User-Friendly Interface:**  Provides an intuitive web interface for seamless image uploads and result display.
* **Crystal-Clear Results:** Displays a clear authorization status with the match percentage. 💯
* **Helpful Feedback:** Offers guidance on improving image quality for optimal results.


## Installation ⬇️

This project needs a Python environment with Flask and a few essential libraries.  A `requirements.txt` file (coming soon!) will list all Python dependencies. Frontend dependencies are linked via CDN in `base.html`.

**To set up the project locally (Instructions coming soon!):**

1.  Clone the repository.
2.  Create a virtual environment: `python3 -m venv venv`
3.  Activate the virtual environment: `source venv/bin/activate` (Linux/macOS) or `venv\Scripts\activate` (Windows).
4.  Install dependencies: `pip install -r requirements.txt`  (Once `requirements.txt` is available!)
5.  Run the Flask application: `flask run`


## Usage 💻

1.  Navigate to the app's URL in your browser.
2.  Head to the "Predict" page.
3.  Upload clear images of your face, iris, and fingerprint.
4.  Click "Predict Now".
5.  The system analyzes the images and displays the authorization status and match percentage.


## How It Works ⚙️

```mermaid
graph LR
    A[User Uploads Images] --> B{Image Preprocessing};
    B --> C[Face Detection (ResNet50 & ViT)];
    B --> D[Iris Detection (Wavelet & ViT)];
    B --> E[Fingerprint Detection (Custom CNN)];
    C --> F[Feature Extraction];
    D --> F;
    E --> F;
    F --> G{Matching & Comparison};
    G -- Authorized --> H[Authorization Success];
    G -- Not Authorized --> I[Authorization Failure];
    H --> J[Display Result];
    I --> J;
```



