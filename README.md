# ChessVision GUI: All-in-One Analysis & Recognition Tool

A robust Python-based Chess GUI that combines Computer Vision (YOLOv8 + OpenCV) with the Stockfish engine. This tool allows users to upload a screenshot of a chessboard, automatically recognize the pieces, generate a FEN string, and analyze the position immediately using an interactive Drag-and-Drop interface.

![Project Status](https://img.shields.io/badge/Status-Active-brightgreen)
![Python Version](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/License-MIT-orange)

## 🌟 Key Features

* **🤖 Neural Board Recognition:** Uses **YOLOv8** to detect pieces and board state from screenshots with high accuracy.
* **👁️ OpenCV Fallback:** Includes a robust heuristic fallback algorithm if neural detection fails.
* **🧠 Stockfish Integration:** Built-in engine analysis with an evaluation bar, best move suggestions, and mate detection.
* **🖱️ Interactive GUI:** Fully functional chessboard built with **PyQt5** supporting drag-and-drop, move validation, and promotion.
* **🔄 Smart Utilities:**
    * **Auto-Crop:** Automatically finds and crops the board from a full-screen screenshot.
    * **Turn Toggle:** Manually switch between White/Black to play if the screenshot creates an ambiguous turn.
    * **Game Modes:** Human vs. Human, Human vs. Engine, or Engine vs. Engine.
    * **FEN Import/Export:** Auto-generates FEN strings from images.

## 🛠️ Built With

* [Python](https://www.python.org/)
* [PyQt5](https://pypi.org/project/PyQt5/) - For the Graphical User Interface.
* [Ultralytics YOLOv8](https://docs.ultralytics.com/) - For neural object detection.
* [OpenCV](https://opencv.org/) - For image processing and board warping.
* [Python-Chess](https://python-chess.readthedocs.io/) - For move generation and validation.
* [Stockfish](https://stockfishchess.org/) - The world's strongest chess engine.

## 🚀 Getting Started

### Prerequisites

* Python 3.8 or higher.
* A valid `stockfish.exe` binary.
* A trained YOLOv8 model (`best.pt`) for chess piece detection.

### Installation

1.  **Clone the repository**
    ```bash
    git clone [https://github.com/yourusername/chess-vision-gui.git](https://github.com/yourusername/chess-vision-gui.git)
    cd chess-vision-gui
    ```

2.  **Install Dependencies**
    ```bash
    pip install ultralytics opencv-python-headless chess stockfish PyQt5 numpy pillow
    ```

3.  **Setup Stockfish & Model**
    * Download Stockfish from [stockfishchess.org](https://stockfishchess.org/download/).
    * Place your `stockfish.exe` and your trained `best.pt` file in the project folder.

## ⚙️ Configuration

Open `all_in_one_chess_gui.py` and update the paths at the top of the file to match your system:

```python
# Configuration Section
STOCKFISH_PATH = r"C:\Path\To\Your\stockfish.exe"
STOCKFISH_DEPTH = 18
MODEL_PATH = "best.pt"
