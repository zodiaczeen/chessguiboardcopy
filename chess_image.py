# all_in_one_chess_gui.py
"""
All-in-One Chess Image GUI (modified)
- Neural piece recognition (optional) + OpenCV fallback
- Image quality checks
- Autocorrect heuristics
- PyQt5 GUI with:
  - Load image at start
  - Render board (SVG->PNG via cairosvg or Qt)
  - Flip board
  - Mode selection (you vs you / you vs engine / engine vs engine)
  - Move input (SAN or UCI)
  - Engine replies (Stockfish) - threaded
  - Undo (single-step)
  - Eval bar + numeric eval + best move (single) (toggleable)
  - True drag-and-drop (legal moves enforced)
Notes:
- Commentary removed.
- Top-3 moves removed; UI kept and shows only ONE best move in the requested format:
    Best Move:
    e2e4    +0.45
- Evaluation & best move updated only after:
    - human move
    - engine move
    - undo
    - load image
- The periodic timer update has been disabled to avoid spamming the engine.
"""
from ultralytics import YOLO
import chess
import chess.svg
from stockfish import Stockfish
import sys
import os
import io
import traceback
import time
from functools import partial
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


# --- ADD THIS IMPORT ---
from ultralytics import YOLO

# --- UPDATE CONFIGURATION ---
# ... (existing config lines) ...
STOCKFISH_PATH = r"C:\Users\ayush\OneDrive\Desktop\project gama\stockfish.exe" 
STOCKFISH_DEPTH = 18

# --- ADD THIS BLOCK ---
stockfish = None
try:
    from stockfish import Stockfish
    if os.path.exists(STOCKFISH_PATH):
        stockfish = Stockfish(path=STOCKFISH_PATH, depth=STOCKFISH_DEPTH)
        print(f"[Stockfish] Loaded successfully from {STOCKFISH_PATH}")
    else:
        print(f"[Stockfish] Warning: Executable not found at {STOCKFISH_PATH}")
except Exception as e:
    print(f"[Stockfish] Initialization failed: {e}")
    stockfish = None
# ----------------------

# Point this to your new YOLO .pt file
MODEL_PATH = "best.pt" 

NEURAL_CONFIDENCE_THRESHOLD = 0.25  # YOLO is usually confident, 0.5 is safe
WARP_SIZE = 800  # We will still warp the board to a flat square
CLASS_ORDER_DEFAULT = ['P','N','B','R','Q','K','p','n','b','r','q','k','empty']


# PyQt5
from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QLabel, QLineEdit, QTextEdit,
    QVBoxLayout, QHBoxLayout, QFileDialog, QMessageBox, QProgressBar, QComboBox,
    QFrame, QDialog, QGridLayout, QRadioButton, QButtonGroup
)
from PyQt5.QtGui import QPixmap, QImage, QPainter, QColor
from PyQt5.QtCore import Qt, QTimer, QPoint

# Optional cairosvg for svg->png rendering
CAIROSVG_AVAILABLE = False
try:
    import cairosvg
    CAIROSVG_AVAILABLE = True
except Exception:
    CAIROSVG_AVAILABLE = False

# Optional TTS lazy init
TTS_AVAILABLE = False
tts_engine = None
def ensure_tts():
    global TTS_AVAILABLE, tts_engine
    if tts_engine is not None:
        return True
    try:
        import pyttsx3
        tts_engine = pyttsx3.init()
        TTS_AVAILABLE = True
        return True
    except Exception as e:
        print("[TTS] init failed:", e)
        TTS_AVAILABLE = False
        return False
def auto_crop_board(img):
    
    if img is None:
        print("Image not found:", img_path)
        return

    h, w = img.shape[:2]

    # Convert to HSV → wood colors have distinct hue
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # MASK FOR WOOD (light + dark squares)
    # These ranges catch both light and dark wooden tiles
    lower = np.array([5, 10, 50])
    upper = np.array([25, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)

    # Morph clean
    kernel = np.ones((7,7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, 3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, 2)

    # Find contours
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not cnts:
        print("Board not detected.")
        return

    # Largest wooden area = chessboard
    cnt = max(cnts, key=cv2.contourArea)
    x, y, w_b, h_b = cv2.boundingRect(cnt)

    board = img[y:y+h_b, x:x+w_b]

    # Show final cropped board
    H, W = img.shape[:2]
    h, w = board.shape[:2]

    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    y0 = (H - h) // 2
    x0 = (W - w) // 2

    return board

    

    




def speak_text(text):
    if not ensure_tts():
        return
    try:
        tts_engine.say(text)
        tts_engine.runAndWait()
    except Exception:
        traceback.print_exc()

# Optional PyTorch neural support (unchanged)
TORCH_AVAILABLE = False
try:
    import torch
    from torchvision import transforms, models
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

# -------------------------
# Configuration
# -------------------------

# Thread pool for background tasks (engine / model)
EXECUTOR = ThreadPoolExecutor(max_workers=3)

# -------------------------
# Utilities: image quality, detection
# -------------------------
# -------------------------
# Engine helpers (top N moves)
# -------------------------
def get_top_n_moves_with_eval(board, n=1):
    """
    Return up to n moves as list of tuples (uci_move, score_str).
    Score string is like "+0.45" or "mate 3".
    """
    out = []
    global stockfish
    try:
        if stockfish is None:
            return out
        top = None
        if hasattr(stockfish, "get_top_moves"):
            try:
                top = stockfish.get_top_moves(n)
            except Exception:
                top = None
        if isinstance(top, list) and top:
            for t in top[:n]:
                mv = t.get("Move") or t.get("move") or t.get("uci") or t
                cp = t.get("Centipawn") or t.get("cp") or t.get("Centipawns")
                if cp is not None:
                    out.append((mv, f"{cp/100:+.2f}"))
                else:
                    # maybe mate
                    if t.get("Mate") is not None:
                        out.append((mv, f"mate {t.get('Mate')}"))
                    else:
                        out.append((mv, str(t)))
            if out:
                return out
    except Exception:
        pass

    # fallback: evaluate each legal move (slower)
    try:
        original_fen = board.fen()
        temp = []
        for mv in board.legal_moves:
            uci = mv.uci()
            try:
                stockfish.set_fen_position(original_fen)
                stockfish.make_moves_from_current_position([uci])
                ev = stockfish.get_evaluation()
                stockfish.set_fen_position(original_fen)

                if ev and isinstance(ev, dict) and ev.get("type") == "cp":
                    score = ev["value"] / 100.0
                    temp.append((uci, f"{score:+.2f}"))
                elif ev and isinstance(ev, dict) and ev.get("type") == "mate":
                    temp.append((uci, f"mate {ev['value']}"))
                else:
                    temp.append((uci, str(ev)))
            except Exception:
                continue

        # sort by numeric value (highest first)
        def score_fn(x):
            mv, sc = x
            if sc.startswith("+") or sc.startswith("-"):
                try:
                    return float(sc)
                except:
                    return 0.0
            if "mate" in sc:
                try:
                    m = int(sc.split()[-1])
                    return 10000 if m > 0 else -10000
                except:
                    return 0.0
            return 0.0

        temp.sort(key=score_fn, reverse=True)
        return temp[:n]
    except Exception:
        return out

# Image quality validation
def validate_image_quality(img_bgr, min_sharpness=120.0, min_board_area_ratio=0.20):
    try:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    except Exception:
        return False, "Image read failed."
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if lap_var < min_sharpness:
        return False, f"Image too blurry (laplacian={lap_var:.1f}). Please retake a clearer photo."
    h, w = gray.shape[:2]
    total_area = h * w
    edges = cv2.Canny(gray, 40, 120)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    max_quad_area = 0
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.03 * peri, True)
        if len(approx) == 4:
            area = cv2.contourArea(approx)
            if area > max_quad_area:
                max_quad_area = area
    board_ratio = max_quad_area / float(total_area) if total_area>0 else 0.0
    if board_ratio < min_board_area_ratio:
        return False, f"Board too small in image (area ratio={board_ratio:.2f}). Crop closer to the chessboard."
    return True, "OK"

# -------------------------
# Neural model loader (optional)
# -------------------------
MODEL = None
CLASS_NAMES = CLASS_ORDER_DEFAULT
transform_square = None
def load_neural_model(path=MODEL_PATH):
    global MODEL
    if not os.path.exists(path):
        print(f"[neural] checkpoint '{path}' not found. Neural disabled.")
        return None
    
    try:
        print(f"[neural] Loading YOLO model from {path}...")
        model = YOLO(path)
        print("[neural] YOLO loaded successfully.")
        print(f"[neural] Classes: {model.names}") # Print classes to check names
        return model
    except Exception as e:
        print(f"[neural] Failed to load YOLO: {e}")
        return None
# -------------------------
# Board detection + warp + squares
# -------------------------
def detect_and_warp_board(img_bgr, out_size=WARP_SIZE):
    img = img_bgr.copy()
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blur, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    quad = None
    max_area = 0
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            area = cv2.contourArea(approx)
            if area > max_area:
                max_area = area
                quad = approx
    if quad is None or max_area < (w*h*0.05):
        src = np.array([[0,0],[w-1,0],[w-1,h-1],[0,h-1]], dtype="float32")
    else:
        pts = quad.reshape(4,2)
        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1).reshape(4)
        tl = pts[np.argmin(s)]
        br = pts[np.argmax(s)]
        tr = pts[np.argmin(diff)]
        bl = pts[np.argmax(diff)]
        src = np.array([tl,tr,br,bl], dtype="float32")
    dst = np.array([[0,0],[out_size-1,0],[out_size-1,out_size-1],[0,out_size-1]], dtype="float32")
    M = cv2.getPerspectiveTransform(src, dst)
    warp = cv2.warpPerspective(img, M, (out_size, out_size))
    return warp

def split_to_squares(warped):
    s = warped.shape[0] // 8
    squares = []
    for r in range(8):
        for c in range(8):
            y0, y1 = r*s, (r+1)*s
            x0, x1 = c*s, (c+1)*s
            sq = warped[y0:y1, x0:x1]
            squares.append(sq)
    return squares

# Build FEN from per-square labels: tries both rank orders
def build_fen_from_labels(labels):
    if len(labels) != 64:
        raise ValueError("labels must be length 64")
    rows = []
    for r in range(8):
        empty = 0
        row = ""
        for f in range(8):
            idx = r*8 + f
            cls = labels[idx]
            if cls == "empty":
                empty += 1
            else:
                if empty:
                    row += str(empty)
                    empty = 0
                row += cls
        if empty:
            row += str(empty)
        rows.append(row)
    cand1 = "/".join(rows) + " w KQkq - 0 1"
    try:
        b1 = chess.Board(cand1)
        if any(p.symbol() == 'K' for p in b1.piece_map().values()) and any(p.symbol() == 'k' for p in b1.piece_map().values()):
            return b1.fen()
    except Exception:
        pass
    cand2 = "/".join(rows[::-1]) + " w KQkq - 0 1"
    try:
        b2 = chess.Board(cand2)
        if any(p.symbol() == 'K' for p in b2.piece_map().values()) and any(p.symbol() == 'k' for p in b2.piece_map().values()):
            return b2.fen()
    except Exception:
        pass
    return cand1

def map_yolo_class_to_fen(class_name):
    # --- 1. HANDLE NEW DATASET (Single Letters) ---
    # The new Roboflow dataset returns 'P', 'n', 'b', etc.
    # We must return them AS IS because Case determines Color.
    # We also ignore the 'board' class.
    valid_fen_chars = ['P', 'N', 'B', 'R', 'Q', 'K', 'p', 'n', 'b', 'r', 'q', 'k']
    
    if class_name in valid_fen_chars:
        return class_name
    
    # If the model detects the whole "board" box, ignore it.
    if class_name == 'board':
        return None
    # -----------------------------------------------

    # --- 2. FALLBACK FOR OLDER MODELS (Long names) ---
    name = class_name.lower().replace(" ", "").replace("-", "_")
    
    if "white" in name:
        if "pawn" in name: return 'P'
        if "rook" in name: return 'R'
        if "knight" in name: return 'N'
        if "bishop" in name: return 'B'
        if "queen" in name: return 'Q'
        if "king" in name: return 'K'
    elif "black" in name:
        if "pawn" in name: return 'p'
        if "rook" in name: return 'r'
        if "knight" in name: return 'n'
        if "bishop" in name: return 'b'
        if "queen" in name: return 'q'
        if "king" in name: return 'k'
    
    # Common abbreviations
    mapping = {
        'wp': 'P', 'wr': 'R', 'wn': 'N', 'wb': 'B', 'wq': 'Q', 'wk': 'K',
        'bp': 'p', 'br': 'r', 'bn': 'n', 'bb': 'b', 'bq': 'q', 'bk': 'k'
    }
    return mapping.get(name, None)

def neural_predict_fen(img_bgr, model):
    if model is None:
        raise RuntimeError("Neural model not provided")
    
    # 1. Warp the board to a standard size (e.g. 800x800)
    # This removes perspective so the grid is perfect squares.
    warp = detect_and_warp_board(img_bgr, out_size=WARP_SIZE)
    
    # 2. Run YOLO inference on the warped image
    results = model(warp, verbose=False)
    
    # Prepare an empty 8x8 grid
    # board_grid[row][col] = (fen_char, confidence)
    board_grid = [[("empty", 0.0) for _ in range(8)] for _ in range(8)]
    
    square_size = WARP_SIZE // 8  # e.g. 100 pixels
    
    # 3. Process Detections
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Get coordinates and class
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0].cpu().numpy())
            cls_id = int(box.cls[0].cpu().numpy())
            class_name = model.names[cls_id]
            
            # Map class name to FEN (P, n, k, etc.)
            fen_char = map_yolo_class_to_fen(class_name)
            if not fen_char:
                continue
            
            # Calculate Center of the piece base
            # We focus slightly lower (y + height*0.8) because chess pieces 
            # stand on the square, but their heads might lean into the square above.
            center_x = (x1 + x2) / 2
            center_y = y2 - (y2 - y1) * 0.15 # Approx feet position
            
            # Map pixel to Grid Indices (0..7)
            col = int(center_x // square_size)
            row = int(center_y // square_size)
            
            # Safety check bounds
            if 0 <= col < 8 and 0 <= row < 8:
                # If a piece is already there, keep the one with higher confidence
                current_piece, current_conf = board_grid[row][col]
                if conf > current_conf:
                     board_grid[row][col] = (fen_char, conf)

    # 4. Build the final FEN string
    # Note: 'detect_and_warp_board' usually puts Rank 8 at the top (row 0)
    # But FEN expects Rank 8 first.
    
    labels = []
    per_sq_flat = []
    
    for r in range(8):
        for c in range(8):
            char, conf = board_grid[r][c]
            labels.append(char)
            per_sq_flat.append((char, conf))
            
    fen = build_fen_from_labels(labels)
    # 1. Warp the board
    warp = detect_and_warp_board(img_bgr, out_size=WARP_SIZE)
    
    # --- ADD THIS DEBUG BLOCK ---
    cv2.imshow("What the AI Sees", warp)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # ----------------------------

    # 2. Run YOLO inference
    results = model(warp, verbose=False)
    # ... (rest of code)
    
    return fen, per_sq_flat, warp

def fallback_opencv_fen(img_bgr):
    warp = detect_and_warp_board(img_bgr, out_size=WARP_SIZE)
    squares = split_to_squares(warp)
    labels = []
    per_sq = []
    for sq in squares:
        gray = cv2.cvtColor(sq, cv2.COLOR_BGR2GRAY)
        
        # Apply a binary threshold to isolate pieces
        # We assume pieces are generally darker or lighter than the square
        # This helps in removing wood grain or shadows
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Calculate the percentage of "on" pixels (the piece)
        pixel_ratio = np.count_nonzero(thresh) / (thresh.shape[0] * thresh.shape[1])

        # --- THIS IS THE CORRECTED LOGIC ---
        # If more than ~10% of the square is filled, assume it's a piece
        # (This threshold needs tuning)
        if pixel_ratio > 0.10: 
            # FALLBACK GUESS: We can't know the piece.
            # We'll just put a pawn. This is a *huge* assumption.
            # A better fallback would check color.
            labels.append("p") 
            per_sq.append(("p", 0.5)) # 'p' is just a placeholder
        else:
            labels.append("empty")
            per_sq.append(("empty", 0.9))
            
    fen = build_fen_from_labels(labels)
    return fen, per_sq, warp

def chesscog_predict_fen(img_bgr):
    try:
        return fallback_opencv_fen(img_bgr)
    except Exception as e:
        print("[chesscog] error:", e)
        return None, None, None

# -------------------------
# Autocorrect heuristics
# -------------------------
def autocorrect_fen(fen, per_square=None):
    try:
        b = chess.Board(fen)
        return b.fen()
    except Exception:
        pass
    if per_square:
        mat = [[per_square[r*8 + c][0] for c in range(8)] for r in range(8)]
        confs = [[per_square[r*8 + c][1] for c in range(8)] for r in range(8)]
        for r in (0,7):
            for c in range(8):
                if mat[r][c] in ('P','p') and confs[r][c] < 0.65:
                    mat[r][c] = 'empty'
        fen_ranks = []
        for r in range(8):
            empty = 0
            rank = ""
            for c in range(8):
                val = mat[r][c]
                if val == 'empty':
                    empty += 1
                else:
                    if empty:
                        rank += str(empty); empty = 0
                    rank += val
            if empty:
                rank += str(empty)
            fen_ranks.append(rank)
        fen_try = "/".join(fen_ranks) + " w KQkq - 0 1"
        try:
            b = chess.Board(fen_try)
            return b.fen()
        except Exception:
            pass
    print("[autocorrect] fallback to startpos")
    return chess.Board().fen()

# -------------------------
# Engine helper (threaded)
# -------------------------
def engine_get_top_n_moves_with_eval(engine, board, n=1):
    out = []
    if engine is None:
        return out
    try:
        top = None
        if hasattr(engine, "get_top_moves"):
            try:
                top = engine.get_top_moves(n)
            except Exception:
                top = None
        if isinstance(top, list) and top:
            for t in top[:n]:
                mv = t.get('Move') or t.get('move') or t.get('uci') or t
                cp = t.get('Centipawn') or t.get('cp') or t.get('Centipawns')
                if cp is not None:
                    out.append((mv, f"{cp/100:+.2f}"))
                else:
                    out.append((mv, str(t)))
            if out:
                return out
    except Exception:
        pass
    original = board.fen()
    items = []
    for mv in board.legal_moves:
        u = mv.uci()
        try:
            engine.set_fen_position(original)
            engine.make_moves_from_current_position([u])
            ev = engine.get_evaluation()
            engine.set_fen_position(original)
            if isinstance(ev, dict) and ev.get('type') == 'cp':
                score = ev.get('value', 0) / 100.0
                items.append((u, f"{score:+.2f}"))
            elif isinstance(ev, dict) and ev.get('type') == 'mate':
                items.append((u, f"mate {ev.get('value')}"))
            else:
                items.append((u, str(ev)))
        except Exception:
            continue
    def keyfn(x):
        mv, s = x
        try:
            return float(s)
        except:
            if "mate" in s:
                try:
                    v = int(s.split()[-1])
                    return 10000 if v>0 else -10000
                except:
                    return 0.0
            return 0.0
    items.sort(key=keyfn, reverse=True)
    return items[:n]

# -------------------------
# Material balance helpers (kept; used for potential future UI)
# -------------------------
def piece_value_char(p):
    if p is None: return 0
    c = p.symbol().lower()
    if c == 'p': return 1
    if c == 'n': return 3
    if c == 'b': return 3
    if c == 'r': return 5
    if c == 'q': return 9
    return 0

def material_balance(board):
    s = 0
    for _, piece in board.piece_map().items():
        v = piece_value_char(piece)
        s += v if piece.color == chess.WHITE else -v
    return s

# -------------------------
# SVG rendering helper
# -------------------------
def svg_to_qpixmap(svg_bytes, size_px=480):
    if CAIROSVG_AVAILABLE:
        try:
            png = cairosvg.svg2png(bytestring=svg_bytes, output_width=size_px, output_height=size_px)
            qim = QImage.fromData(png)
            pix = QPixmap.fromImage(qim)
            return pix
        except Exception:
            pass
    try:
        img = QImage.fromData(svg_bytes, "SVG")
        pix = QPixmap.fromImage(img)
        return pix.scaled(size_px, size_px, Qt.KeepAspectRatio, Qt.SmoothTransformation)
    except Exception:
        pass
    return None

# -------------------------
# Promotion dialog (used during drag promotion)
# -------------------------
class PromotionDialog(QDialog):
    def __init__(self, parent=None, color=chess.WHITE):
        super().__init__(parent)
        self.setWindowTitle("Choose promotion piece")
        self.setModal(True)
        self.choice = 'q'
        layout = QVBoxLayout()
        grid = QHBoxLayout()
        self.group = QButtonGroup(self)
        pieces = ['q','r','b','n']
        labels = ['Queen','Rook','Bishop','Knight']
        for i,p in enumerate(pieces):
            rb = QRadioButton(labels[i])
            if i == 0:
                rb.setChecked(True)
            self.group.addButton(rb, i)
            grid.addWidget(rb)
        layout.addLayout(grid)
        ok = QPushButton("OK")
        ok.clicked.connect(self.accept_choice)
        layout.addWidget(ok)
        self.setLayout(layout)
    def accept_choice(self):
        idx = self.group.checkedId()
        mapping = {0:'q',1:'r',2:'b',3:'n'}
        self.choice = mapping.get(idx, 'q')
        self.accept()

# -------------------------
# BoardWidget with drag-and-drop (legal moves only)
# -------------------------
class BoardWidget(QLabel):
    """
    Interactive board widget:
    - Displays board pixmap
    - Handles true drag-and-drop of pieces
    - Enforces legal moves
    - Uses cropped theme-specific square images for drag icon
    """

    def __init__(self, app, parent=None):
        super().__init__(parent)
        self.app = app  # parent ChessImageApp
        self.setFixedSize(480, 480)
        self.setStyleSheet("background:#111; border:2px solid #333;")

        # Drag state
        self.dragging = False
        self.drag_label = None
        self.drag_pix = None
        self.drag_src_sq = None
        self.square_px = 60  # updated dynamically

    # ------------------------------------------------------------------
    # Utility: update square size based on widget size
    # ------------------------------------------------------------------
    def update_square_size(self):
        self.square_px = min(self.width(), self.height()) // 8

    # ------------------------------------------------------------------
    # Convert pixel → chess square index (0..63)
    # ------------------------------------------------------------------
    def _pixel_to_square(self, x, y):
        self.update_square_size()
        if x < 0 or y < 0 or x >= self.width() or y >= self.height():
            return None

        file_vis = int(x // self.square_px)  # col index
        row_vis = int(y // self.square_px)   # row index

        # Normal orientation: top row is rank 8
        if self.app.board_flipped:
            # board rotated 180 degrees
            chess_file = 7 - file_vis
            chess_rank = row_vis
        else:
            chess_file = file_vis
            chess_rank = 7 - row_vis

        if not (0 <= chess_file <= 7 and 0 <= chess_rank <= 7):
            return None

        return chess.square(chess_file, chess_rank)

    # ------------------------------------------------------------------
    # Drag start
    # ------------------------------------------------------------------
    def mousePressEvent(self, ev):
        if ev.button() != Qt.LeftButton:
            return

        pos = ev.pos()
        sq = self._pixel_to_square(pos.x(), pos.y())
        if sq is None:
            return

        piece = self.app.board.piece_at(sq)
        if piece is None:
            return

        # Get currently displayed board pixmap
        board_pix = self.pixmap()
        if board_pix is None or board_pix.isNull():
            # fallback: re-render
            board_pix = self.app.render_board_pixmap(self.app.board)
            if board_pix is None:
                return

        # Crop the selected square from the board pixmap
        self.update_square_size()
        file_vis = int(pos.x() // self.square_px)
        row_vis = int(pos.y() // self.square_px)
        x0 = file_vis * self.square_px
        y0 = row_vis * self.square_px

        try:
            cropped = board_pix.copy(x0, y0, self.square_px, self.square_px)
        except Exception:
            cropped = board_pix.scaled(self.square_px, self.square_px, Qt.KeepAspectRatio, Qt.SmoothTransformation)

        # Begin drag
        self.dragging = True
        self.drag_src_sq = sq
        self.drag_pix = cropped

        # Drag label follows cursor
        self.drag_label = QLabel(self)
        self.drag_label.setPixmap(self.drag_pix)
        self.drag_label.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.drag_label.setWindowFlags(Qt.SubWindow)
        self.drag_label.resize(self.drag_pix.size())
        self._move_drag_label(ev.pos())
        self.drag_label.show()

    # ------------------------------------------------------------------
    # Drag move
    # ------------------------------------------------------------------
    def mouseMoveEvent(self, ev):
        if self.dragging:
            self._move_drag_label(ev.pos())

    # ------------------------------------------------------------------
    # Drag stop
    # ------------------------------------------------------------------
    def mouseReleaseEvent(self, ev):
        if not self.dragging:
            return

        pos = ev.pos()
        dst_sq = self._pixel_to_square(pos.x(), pos.y())
        src_sq = self.drag_src_sq

        # Cleanup drag label
        self.dragging = False
        if self.drag_label:
            self.drag_label.hide()
            self.drag_label.deleteLater()
        self.drag_label = None
        self.drag_pix = None
        self.drag_src_sq = None

        if dst_sq is None:
            return

        # Build move object
        move = chess.Move(src_sq, dst_sq)

        # Handle promotion
        piece = self.app.board.piece_at(src_sq)
        if piece and piece.piece_type == chess.PAWN and chess.square_rank(dst_sq) in (0, 7):
            dlg = PromotionDialog(self, color=piece.color)
            if dlg.exec_() == QDialog.Accepted:
                prom = dlg.choice
                promo_map = {'q': chess.QUEEN, 'r': chess.ROOK, 'b': chess.BISHOP, 'n': chess.KNIGHT}
                move = chess.Move(src_sq, dst_sq, promotion=promo_map[prom])
            else:
                move = chess.Move(src_sq, dst_sq, promotion=chess.QUEEN)

        # If legal → play it
        if move in self.app.board.legal_moves:
            self.app.history.append(self.app.board.fen())
            self.app.board.push(move)
            self.app.on_after_user_move()
        else:
            return

    # ------------------------------------------------------------------
    # Move drag icon
    # ------------------------------------------------------------------
    def _move_drag_label(self, pos):
        if not self.drag_label or self.drag_pix is None:
            return
        x = pos.x() - self.drag_pix.width() // 2
        y = pos.y() - self.drag_pix.height() // 2
        self.drag_label.move(x, y)

# -------------------------
# ChessImageApp — Full GUI
# -------------------------
class ChessImageApp(QWidget):
    global stockfish

    def __init__(self, model_tuple=None):
        super().__init__()
        self.setWindowTitle("All-in-One Chess GUI (Drag & Drop + Image Detection)")
        self.setStyleSheet("background:#1a1a1a; color:#eaeaea; font-family:Arial;")

        self.model_tuple = model_tuple
        self.board = chess.Board()
        self.history = []
        self.board_flipped = False

        # Engine feature toggles (OFF by default)
        self.engine_enabled = False
        self.show_eval = False
        # show_top_moves replaced with single best move concept
        self.show_best_move = False

        self.tts_enabled = False

        # --- Widgets ---
        self.board_widget = BoardWidget(app=self)

        # Eval label
        self.eval_label = QLabel("Eval: ---")
        self.eval_label.setStyleSheet("font-size:16px; color:#00e5ff;")
        self.eval_label.hide()

        # Eval bar
        self.eval_bar = QProgressBar()
        self.eval_bar.setRange(-300, 300)
        self.eval_bar.setTextVisible(False)
        self.eval_bar.hide()
        self.eval_bar.setStyleSheet("""
            QProgressBar { border: 1px solid #333; background: #0b0b0b; height: 16px; }
            QProgressBar::chunk { background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
                stop:0 #00ff9c, stop:1 #00d0ff); }
        """)

        # Best move panel (we keep the box UI but it shows only the chosen format)
        self.top3_box = QTextEdit()
        self.top3_box.setReadOnly(True)
        self.top3_box.hide()
        self.top3_box.setFixedHeight(80)
        self.top3_box.setStyleSheet("background:#080808; color:#9fff7f; font-family:Consolas;")

        # Mode selection
        self.mode_combo = QComboBox()
        self.mode_combo.addItems([
            "You both sides",
            "You play White (engine replies as Black)",
            "You play Black (engine replies as White)",
            "Engine vs Engine"
        ])
        self.mode_combo.currentIndexChanged.connect(self.on_mode_change)

        # Load image
        self.load_btn = QPushButton("Load Image")
        self.load_btn.clicked.connect(self.on_load_image)

        # Move entry
        self.move_entry = QLineEdit()
        self.move_entry.setPlaceholderText("Enter move (e4, Nf3, e2e4...)")
        self.move_entry.setStyleSheet("padding:6px; background:#111; color:#fff;")

        self.play_btn = QPushButton("Play Move")
        self.play_btn.clicked.connect(self.on_play_move)

        self.undo_btn = QPushButton("Undo")
        self.undo_btn.clicked.connect(self.on_undo)

        self.flip_btn = QPushButton("Flip Board")
        self.flip_btn.clicked.connect(self.on_flip)

        self.engine_btn = QPushButton("Engine: Play Best")
        self.engine_btn.clicked.connect(self.on_engine_play)
        self.engine_btn.hide()  # hidden unless engine enabled

        # TTS toggle
        self.tts_btn = QPushButton("TTS: Off")
        self.tts_btn.setCheckable(True)
        self.tts_btn.clicked.connect(self.on_toggle_tts)
        self.tts_btn.hide()

        # Engine master toggle
        self.engine_master_btn = QPushButton("Enable Engine Features")
        self.engine_master_btn.clicked.connect(self.toggle_engine_master)

        # --- Layout ---
        left = QVBoxLayout()
        left.addWidget(self.board_widget)
        left.addWidget(self.eval_label)
        left.addWidget(self.eval_bar)

        right = QVBoxLayout()
        right.addWidget(QLabel("Mode:"))
        right.addWidget(self.mode_combo)
        right.addWidget(self.load_btn)

        right.addWidget(self.move_entry)
        row1 = QHBoxLayout()
        row1.addWidget(self.play_btn)
        row1.addWidget(self.undo_btn)
        right.addLayout(row1)

        row2 = QHBoxLayout()
        row2.addWidget(self.flip_btn)
        row2.addWidget(self.engine_btn)
        right.addLayout(row2)

        right.addWidget(self.engine_master_btn)
        right.addWidget(self.tts_btn)
        right.addWidget(QLabel("Best Move:"))
        right.addWidget(self.top3_box)

        main = QHBoxLayout()
        main.addLayout(left)
        main.addLayout(right)

        self.setLayout(main)
        self.resize(1000, 600)

        # Timer removed/disabled to avoid periodic engine spamming.
        # If you want periodic updates later, re-enable carefully.
        self.update_timer = QTimer()
        # do NOT start the timer

        self.refresh_board()

    # --------------------------------------------------------------
    # After user plays a move (by drag or text entry)
    # --------------------------------------------------------------
    def on_after_user_move(self):
        self.refresh_board()
        # Update eval & best move after the move
        self.update_engine_info()

        idx = self.mode_combo.currentIndex()
        if not self.engine_enabled:
            return

        # Engine replies automatically based on mode
        if idx == 1 and self.board.turn == chess.BLACK:
            QTimer.singleShot(200, self.engine_play_move)
        elif idx == 2 and self.board.turn == chess.WHITE:
            QTimer.singleShot(200, self.engine_play_move)
        elif idx == 3:
            QTimer.singleShot(200, self.engine_play_move)

    # --------------------------------------------------------------
    # Refresh board rendering
    # --------------------------------------------------------------
    def refresh_board(self):
        pix = self.render_board_pixmap(self.board)
        if pix:
            self.board_widget.setPixmap(pix)

    def render_board_pixmap(self, board_obj):
        try:
            orient = chess.BLACK if self.board_flipped else chess.WHITE
            svg = chess.svg.board(board=board_obj, size=480, orientation=orient).encode("utf-8")
            return svg_to_qpixmap(svg, size_px=480)
        except:
            traceback.print_exc()
            return None

    # --------------------------------------------------------------
    # Load image → detect → build FEN
    # --------------------------------------------------------------
    def on_load_image(self):
        fname, _ = QFileDialog.getOpenFileName(self, "Select Chessboard Image", "", "Images (*.png *.jpg *.jpeg)")
        if not fname:
            return

        img = cv2.imread(fname)
        if img is None:
            return

        # --- FIX: Attempt to crop the board automatically ---
        cropped = auto_crop_board(img)
        if cropped is not None:
            img = cropped
            print("[Info] Board auto-cropped successfully.")
        else:
            print("[Warning] Could not auto-crop. Using full image.")
        # ----------------------------------------------------

        ok, msg = validate_image_quality(img)
        if not ok:
            # We allow it to proceed but warn the user
            QMessageBox.warning(self, "Image Quality Warning", msg)

        # Try neural → fallback
        fen = None
        per_sq = None
        used_neural = False

        # Note: self.model_tuple is actually just the YOLO model object now
        if self.model_tuple is not None:
            try:
                fen_pred, per_sq, warp = neural_predict_fen(img, self.model_tuple)
                # Calculate average confidence
                conf = np.mean([c for _, c in per_sq]) if per_sq else 0.0
                
                print(f"[Neural] Prediction finished. Avg Confidence: {conf:.2f}")
                
                if conf >= NEURAL_CONFIDENCE_THRESHOLD:
                    fen = fen_pred
                    used_neural = True
            except Exception as e:
                print(f"[Neural] Error during prediction: {e}")
                traceback.print_exc()

        if not used_neural:
            print("[System] Falling back to OpenCV detection...")
            fen2, per_sq2, warp2 = chesscog_predict_fen(img)
            if fen2 is not None:
                fen = fen2
                per_sq = per_sq2
            else:
                QMessageBox.critical(self, "Error", "Recognition failed.")
                return

        # Autocorrect and Load
        fen = autocorrect_fen(fen, per_sq if per_sq else None)
        try:
            self.board = chess.Board(fen)
        except Exception:
            self.board = chess.Board()

        self.history = []
        self.refresh_board()
        self.update_engine_info()
        
        source = "YOLO AI" if used_neural else "OpenCV Fallback"
        QMessageBox.information(self, "Loaded", f"Position loaded using {source}.")
    # --------------------------------------------------------------
    # Engine master toggle
    # --------------------------------------------------------------
    def toggle_engine_master(self):
        self.engine_enabled = not self.engine_enabled
        if self.engine_enabled:
            self.engine_master_btn.setText("Disable Engine Features")
            self.engine_btn.show()
            self.eval_bar.show()
            self.eval_label.show()
            self.top3_box.show()
            self.tts_btn.show()
            self.show_eval = True
            self.show_best_move = True
        else:
            self.engine_master_btn.setText("Enable Engine Features")
            self.engine_btn.hide()
            self.eval_bar.hide()
            self.eval_label.hide()
            self.top3_box.hide()
            self.tts_btn.hide()
            self.show_eval = False
            self.show_best_move = False

        # Update display when toggling
        self.update_engine_info()

    # --------------------------------------------------------------
    # Manual play move (text input)
    # --------------------------------------------------------------
    def on_play_move(self):
        mv = self.move_entry.text().strip()
        if not mv:
            return
        self.move_entry.clear()

        try:
            try:
                move = self.board.parse_san(mv)
            except:
                move = chess.Move.from_uci(mv)

            if move not in self.board.legal_moves:
                QMessageBox.warning(self, "Illegal Move", "Move not legal.")
                return

            self.history.append(self.board.fen())
            self.board.push(move)
            self.on_after_user_move()

        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    # --------------------------------------------------------------
    # Engine move
    # --------------------------------------------------------------
    def get_engine_move(self):
        try:
            if stockfish is None:
                return None
            stockfish.set_fen_position(self.board.fen())
            # Some Stockfish wrappers use get_best_move or get_best_move_time etc.
            if hasattr(stockfish, "get_best_move"):
                return stockfish.get_best_move()
            elif hasattr(stockfish, "get_best_move_time"):
                return stockfish.get_best_move_time(100)  # small time fallback
            else:
                # try attribute name commonly used
                return stockfish.get_best_move()
        except Exception:
            return None

    def engine_play_move(self):
        mv = self.get_engine_move()
        if not mv:
            return
        try:
            self.history.append(self.board.fen())
            self.board.push(chess.Move.from_uci(mv))
            self.on_after_user_move()
        except Exception:
            # if engine returned an invalid move for current board, ignore
            traceback.print_exc()

    def on_engine_play(self):
        self.engine_play_move()

    # --------------------------------------------------------------
    # Undo
    # --------------------------------------------------------------
    def on_undo(self):
        if not self.history:
            return
        last = self.history.pop()
        self.board = chess.Board(last)
        self.refresh_board()
        # Update eval & best move after undo
        self.update_engine_info()

    # --------------------------------------------------------------
    # Flip
    # --------------------------------------------------------------
    def on_flip(self):
        self.board_flipped = not self.board_flipped
        self.refresh_board()

    # --------------------------------------------------------------
    # Mode change (forces flip or normal orientation)
    # --------------------------------------------------------------
    def on_mode_change(self, idx):
        # idx:
        # 0 = you both sides
        # 1 = you play white
        # 2 = you play black
        # 3 = engine vs engine

        # Auto flip when playing Black
        if idx == 2:
            self.board_flipped = True
        else:
            self.board_flipped = False

        self.refresh_board()

    # --------------------------------------------------------------
    # TTS toggle
    # --------------------------------------------------------------
    def on_toggle_tts(self):
        self.tts_enabled = self.tts_btn.isChecked()
        self.tts_btn.setText("TTS: On" if self.tts_enabled else "TTS: Off")

    # --------------------------------------------------------------
    # Engine info update (eval, best move)
    # This is called only when a real change happens (move, engine move, undo, load)
    # --------------------------------------------------------------
    def update_engine_info(self):
        # Only update when engine features are enabled
        if not self.engine_enabled or stockfish is None:
            # clear/hide fields
            self.eval_label.setText("Eval: ---")
            self.eval_bar.setValue(0)
            if not self.top3_box.isHidden():
                self.top3_box.setText("Best Move:\n---")
            return

        try:
            stockfish.set_fen_position(self.board.fen())
            ev = None
            try:
                ev = stockfish.get_evaluation()
            except Exception:
                ev = None

            # Eval
            if self.show_eval and ev:
                if isinstance(ev, dict) and ev.get("type") == "cp":
                    cp = ev.get("value", 0) / 100.0
                    self.eval_label.setText(f"Eval: {cp:+.2f}")
                    clamped = max(-300, min(300, int(ev.get('value', 0))))
                    self.eval_bar.setValue(clamped)
                elif isinstance(ev, dict) and ev.get("type") == "mate":
                    self.eval_label.setText(f"Mate in {ev.get('value')}")
                    # Set bar to extreme
                    v = 300 if ev.get("value", 0) > 0 else -300
                    self.eval_bar.setValue(v)
                else:
                    # unknown format
                    self.eval_label.setText("Eval: ---")
                    self.eval_bar.setValue(0)
            else:
                # no eval to show
                self.eval_label.setText("Eval: ---")
                self.eval_bar.setValue(0)

            # Best move (single)
            if self.show_best_move:
                best = get_top_n_moves_with_eval(self.board, 1)
                if best:
                    mv, score = best[0]
                    # Normalize move string for readability
                    # some engines return SAN, some UCI; we keep returned string
                    display_text = f"Best Move:\n{mv:6s}    {score}"
                    self.top3_box.setText(display_text)
                else:
                    self.top3_box.setText("Best Move:\n---")
        except Exception:
            traceback.print_exc()
            self.eval_label.setText("Eval: ---")
            self.eval_bar.setValue(0)
            if not self.top3_box.isHidden():
                self.top3_box.setText("Best Move:\n---")

# -------------------------
# main()
# -------------------------
def main():
    # Attempt to load model independently of the old TORCH_AVAILABLE flag
    model = None
    if os.path.exists(MODEL_PATH):
        model = load_neural_model(MODEL_PATH)
    else:
        print(f"[main] Model file {MODEL_PATH} not found. Using OpenCV fallback.")

    app = QApplication(sys.argv)
    # Pass the model directly (variable name model_tuple is kept for compatibility with init)
    w = ChessImageApp(model_tuple=model)
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
