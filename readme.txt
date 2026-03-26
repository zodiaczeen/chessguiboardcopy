All-in-One Chess Image GUI (Ultra Pro v2)
A powerful, multithreaded desktop application that bridges computer vision and chess analysis. This tool captures chess board configurations from images using a custom-trained YOLOv8 model, visualizes the board in a PyQt5 GUI, and provides real-time Stockfish evaluation with "Lichess-style" analysis arrows.

It also includes a built-in "Editor Mode" to fix vision errors and an "Active Learning" module to save corrected boards for future YOLO retraining.

🚀 Features
Neural Vision: Uses YOLOv8 (best.pt) to detect pieces and generate FEN strings from images.

Pro GUI: Features a vertical Chess.com-style evaluation bar and Lichess-style engine arrows.

Multithreaded: Engine analysis and YOLO inference run on separate threads to keep the UI responsive.

Editor Mode: Manually add, remove, or fix pieces if the computer vision makes a mistake.

Active Learning Saver: Instantly save warped board images and their corrected FEN labels to build a better dataset.

📋 Prerequisites
You will need Python 3.8+ installed on your system. Install the required Python libraries using pip:

Bash
pip install opencv-python numpy chess stockfish ultralytics PyQt5
⚙️ Setup & Configuration
Stockfish Engine:

Download the latest Stockfish engine.

Extract the .exe file.

Open the Python script and update the STOCKFISH_PATH variable to point to your stockfish.exe location.

YOLOv8 Model:

Ensure your trained YOLOv8 weights file (best.pt) is placed in the same root directory as the Python script.

⚠️ Image Capture Constraints (Important)
Because the auto-cropping algorithm currently relies on contour detection (brute force), you must adhere to the following conditions when taking screenshots for the YOLO model to accurately parse the board:

Site: Use Chess.com in full-screen mode.

Board Theme: Set the board theme to Wood.

Piece Theme: Set the piece style to Tournament.

Perspective: Always capture screenshots from the White perspective (White at the bottom).

💻 Usage
Run the application:

Bash
python all_in_one_chess_gui_ultra_v2.py
Click "📂 Load Image" to select your Chess.com screenshot.

The YOLO model will predict the FEN and load the board.

Toggle "Enable Engine" to start Stockfish evaluation.

If the prediction is slightly off, click "✏️ Edit Board" to correct the position, then use "💾 Save Correction" to save the image-FEN pair for future model training.
