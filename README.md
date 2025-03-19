# PythonPose - Background Extraction with OpenCV

## 📌 Description
PythonPose is a quick Python project that utilizes OpenCV to extract backgrounds from images. It includes various scripts to process images, find contours, and identify positions based on different angles.

## 🔧 Installation
1. Clone this repository:
   ```sh
   git clone https://github.com/yourusername/PythonPose.git
   ```
2. Navigate into the project directory:
   ```sh
   cd PythonPose
   ```
3. Install the required dependencies:
   ```sh
   pip install -r requirements.txt
   ```

## 🚀 Usage
To use the background extraction scripts, run:
```sh
python bgEraserAI.py --input path/to/image.jpg --output path/to/output.jpg
```
Replace `path/to/image.jpg` with your actual image file and specify the desired output path.

## 📂 Project Structure
- **ProcessAllSizes.py** - Handles different image sizes for processing.
- **bgEraserAI.py** - Main script for background removal.
- **findContoursFront.py** / **findContoursSide.py** - Finds image contours from different angles.
- **findPositionsFront.py** / **findPositionsSide.py** - Identifies key positions in the image.
- **randomTesting.py** - Script for testing different configurations.
- **Initial Measurements/** - Contains scripts for initial measurement calculations.

## 🛠 Dependencies
- Python 3.x
- OpenCV
- NumPy
- Any other dependencies listed in `requirements.txt`

## ✨ Contributions
Feel free to fork this repository, create feature branches, and submit pull requests!

## 📜 License
This project is open-source and available under the [MIT License](LICENSE).
