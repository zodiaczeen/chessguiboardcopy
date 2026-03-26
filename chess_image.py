# all_in_one_chess_gui_ultra_v2.py
"""
All-in-One Chess Image GUI (Ultra Pro v2)
- "Chess.com" Vertical Evaluation Bar
- "Lichess" Style Analysis Arrows
- "Editor Mode" to fix incorrect boards
- "Active Learning" Data Saver (Save Image + Corrected FEN)
- Multithreaded Engine & Vision
"""
import sys
import os
import math
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import chess
import chess.svg
import chess.pgn
from stockfish import Stockfish
from ultralytics import YOLO

# PyQt5 Imports
from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QLabel, QLineEdit, QTextEdit,
    QVBoxLayout, QHBoxLayout, QFileDialog, QMessageBox, QFrame, 
    QRadioButton, QButtonGroup, QGridLayout, QScrollArea, QToolButton
)
from PyQt5.QtGui import QPixmap, QImage, QPainter, QColor, QPen, QBrush, QPolygonF, QIcon
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QObject, QPoint

# --- CONFIGURATION ---
STOCKFISH_PATH = r"C:\Users\ayush\OneDrive\Desktop\project gama\stockfish.exe"
STOCKFISH_DEPTH = 18
MODEL_PATH = "best.pt" 
WARP_SIZE = 800  

# --- COLORS ---
COLOR_LIGHT = QColor(238, 238, 210)
COLOR_DARK = QColor(118, 150, 86)
COLOR_HIGHLIGHT = QColor(186, 202, 68, 200) 
COLOR_SELECTED = QColor(255, 255, 0, 100)
ARROW_COLORS = [QColor(0, 200, 0, 180), QColor(0, 100, 255, 180), QColor(255, 100, 0, 180)]

# --- UTILS ---
def svg_to_pixmap(svg_bytes, size):
    img = QImage.fromData(svg_bytes)
    return QPixmap.fromImage(img).scaled(size, size, Qt.KeepAspectRatio, Qt.SmoothTransformation)

# --- CUSTOM WIDGETS ---

class PiecePalette(QWidget):
    """Sidebar to select pieces for editing"""
    piece_selected = pyqtSignal(object) # Emits chess.Piece or None (for trash)

    def __init__(self):
        super().__init__()
        self.layout = QGridLayout()
        self.layout.setSpacing(5)
        self.setLayout(self.layout)
        self.current_selection = None
        self.buttons = []
        self.init_ui()

    def init_ui(self):
        # 1. Trash Button (Clear Square)
        btn_trash = QToolButton()
        btn_trash.setText("❌")
        btn_trash.setStyleSheet("font-size: 20px; padding: 5px; background: #442222; color: white;")
        btn_trash.clicked.connect(lambda: self.select_piece(None, btn_trash))
        self.layout.addWidget(btn_trash, 0, 0, 1, 2)
        self.buttons.append(btn_trash)

        # 2. Pieces
        pieces = [
            chess.Piece(chess.PAWN, chess.WHITE), chess.Piece(chess.PAWN, chess.BLACK),
            chess.Piece(chess.KNIGHT, chess.WHITE), chess.Piece(chess.KNIGHT, chess.BLACK),
            chess.Piece(chess.BISHOP, chess.WHITE), chess.Piece(chess.BISHOP, chess.BLACK),
            chess.Piece(chess.ROOK, chess.WHITE), chess.Piece(chess.ROOK, chess.BLACK),
            chess.Piece(chess.QUEEN, chess.WHITE), chess.Piece(chess.QUEEN, chess.BLACK),
            chess.Piece(chess.KING, chess.WHITE), chess.Piece(chess.KING, chess.BLACK),
        ]

        row, col = 1, 0
        for p in pieces:
            btn = QToolButton()
            # Render piece icon
            svg = chess.svg.piece(p, size=40).encode('utf-8')
            pix = svg_to_pixmap(svg, 40)
            btn.setIcon(QIcon(pix))
            btn.setIconSize(pix.size())
            btn.setStyleSheet("background: #333; padding: 5px; border: 1px solid #555;")
            
            # Use closure to capture 'p'
            btn.clicked.connect(lambda checked, piece=p, b=btn: self.select_piece(piece, b))
            
            self.layout.addWidget(btn, row, col)
            self.buttons.append(btn)
            col += 1
            if col > 1:
                col = 0
                row += 1

    def select_piece(self, piece, btn_obj):
        self.current_selection = piece
        self.piece_selected.emit(piece)
        
        # Update Visuals
        for b in self.buttons:
            b.setStyleSheet("background: #333; padding: 5px; border: 1px solid #555;")
        btn_obj.setStyleSheet("background: #007acc; padding: 5px; border: 2px solid white;")

class EvalBarWidget(QWidget):
    """Vertical bar like Chess.com"""
    def __init__(self):
        super().__init__()
        self.setFixedWidth(30)
        self.score = 0.0 
        self.is_mate = False

    def set_eval(self, score_cp, is_mate=False):
        self.score = score_cp
        self.is_mate = is_mate
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        w, h = self.width(), self.height()
        if self.is_mate:
            pct = 1.0 if self.score > 0 else 0.0
        else:
            normalized = max(-1000, min(1000, self.score))
            pct = 0.5 + (normalized / 2000) 
        
        white_h = int(h * pct)
        painter.fillRect(0, 0, w, h, QColor(60, 60, 60)) 
        painter.fillRect(0, h - white_h, w, white_h, QColor(240, 240, 240))
        
        painter.setPen(Qt.black if pct > 0.5 else Qt.white)
        txt = "M" if self.is_mate else f"{abs(self.score)/100:.1f}"
        fm = painter.fontMetrics()
        tw = fm.width(txt)
        if pct > 0.5: 
            painter.drawText((w-tw)//2, h - 10, txt)
        else: 
            painter.drawText((w-tw)//2, 20, txt)

# --- WORKERS ---
class EngineWorker(QObject):
    result_ready = pyqtSignal(list, float, bool)

    def __init__(self, path, depth):
        super().__init__()
        self.path = path
        self.depth = depth
        self.stockfish = None
        self.init_engine()

    def init_engine(self):
        try:
            if os.path.exists(self.path):
                params = {"Threads": 2, "Hash": 64, "Ponder": "true", "Minimum Thinking Time": 20}
                self.stockfish = Stockfish(path=self.path, depth=self.depth, parameters=params)
                self.stockfish.set_skill_level(20)
        except Exception as e: print(f"Engine Error: {e}")

    def analyze(self, fen):
        if not self.stockfish: return
        try:
            self.stockfish.set_fen_position(fen)
            top_moves = self.stockfish.get_top_moves(3)
            eval_data = self.stockfish.get_evaluation()
            val = eval_data.get('value', 0)
            is_mate = eval_data.get('type') == 'mate'
            self.result_ready.emit(top_moves, float(val), is_mate)
        except: pass

class YoloWorker(QThread):
    result_ready = pyqtSignal(str)

    def __init__(self, model_path, img):
        super().__init__()
        self.model_path = model_path
        self.img = img

    def run(self):
        try:
            model = YOLO(self.model_path)
            fen, _, _ = neural_predict_fen(self.img, model)
            self.result_ready.emit(fen)
        except: self.result_ready.emit("")

# --- VISION UTILS ---
def auto_crop_board(img):
    if img is None: return None
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array([5, 10, 50])
    upper = np.array([25, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    kernel = np.ones((7,7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, 3)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return None
    cnt = max(cnts, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(cnt)
    return img[y:y+h, x:x+w]

def detect_and_warp_board(img, out_size=WARP_SIZE):
    # Same as before
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blur, 50, 150)
    cnts, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    quad, max_area = None, 0
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            area = cv2.contourArea(approx)
            if area > max_area: max_area, quad = area, approx
    h, w = img.shape[:2]
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
    return cv2.warpPerspective(img, M, (out_size, out_size))

def map_yolo_to_fen(name):
    valid = ['P', 'N', 'B', 'R', 'Q', 'K', 'p', 'n', 'b', 'r', 'q', 'k']
    if name in valid: return name
    mapping = {'wp':'P','wr':'R','wn':'N','wb':'B','wq':'Q','wk':'K',
               'bp':'p','br':'r','bn':'n','bb':'b','bq':'q','bk':'k'}
    return mapping.get(name.lower().replace("-","_"), None)

def neural_predict_fen(img, model):
    warp = detect_and_warp_board(img)
    results = model(warp, verbose=False)
    grid = [[("empty", 0.0) for _ in range(8)] for _ in range(8)]
    sq_sz = WARP_SIZE // 8
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0])
            cls_name = model.names[int(box.cls[0])]
            fen = map_yolo_to_fen(cls_name)
            if not fen: continue
            cx, cy = (x1+x2)/2, y2 - (y2-y1)*0.15
            c, r_idx = int(cx // sq_sz), int(cy // sq_sz)
            if 0<=c<8 and 0<=r_idx<8:
                if conf > grid[r_idx][c][1]: grid[r_idx][c] = (fen, conf)
    rows = []
    for r in range(8):
        empty, row = 0, ""
        for c in range(8):
            char = grid[r][c][0]
            if char == "empty": empty += 1
            else:
                if empty: row += str(empty); empty=0
                row += char
        if empty: row += str(empty)
        rows.append(row)
    return "/".join(rows) + " w KQkq - 0 1", [], warp

# -------------------------
# PRO BOARD WIDGET (UPDATED FOR EDITING)
# -------------------------
class ChessBoardWidget(QWidget):
    move_made = pyqtSignal(chess.Move)
    board_changed = pyqtSignal() # Emitted when edited

    def __init__(self, board, parent=None):
        super().__init__(parent)
        self.board = board
        self.flipped = False
        self.setFixedSize(600, 600)
        
        # Edit Mode State
        self.edit_mode = False
        self.editor_piece = None # The piece selected in palette
        
        # Interaction State
        self.dragging = False
        self.drag_piece = None
        self.drag_start_sq = None
        self.mouse_pos = QPoint(0,0)
        
        self.engine_arrows = [] 
        self.piece_pixmaps = {} 
        self.cache_pieces(75) 
        self.setMouseTracking(True)

    def cache_pieces(self, size):
        pieces = [
            chess.Piece(chess.PAWN, chess.WHITE), chess.Piece(chess.KNIGHT, chess.WHITE),
            chess.Piece(chess.BISHOP, chess.WHITE), chess.Piece(chess.ROOK, chess.WHITE),
            chess.Piece(chess.QUEEN, chess.WHITE), chess.Piece(chess.KING, chess.WHITE),
            chess.Piece(chess.PAWN, chess.BLACK), chess.Piece(chess.KNIGHT, chess.BLACK),
            chess.Piece(chess.BISHOP, chess.BLACK), chess.Piece(chess.ROOK, chess.BLACK),
            chess.Piece(chess.QUEEN, chess.BLACK), chess.Piece(chess.KING, chess.BLACK)
        ]
        for p in pieces:
            svg = chess.svg.piece(p, size=size).encode('utf-8')
            self.piece_pixmaps[p.symbol()] = svg_to_pixmap(svg, size)

    def set_arrows(self, top_moves):
        self.engine_arrows = []
        for i, move_data in enumerate(top_moves):
            if i >= 3: break
            uci = move_data.get('Move')
            if uci:
                move = chess.Move.from_uci(uci)
                color = ARROW_COLORS[i]
                self.engine_arrows.append((move, color))
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        sq_size = self.width() // 8
        
        # 1. Squares
        for r in range(8):
            for c in range(8):
                is_light = (r + c) % 2 == 0
                color = COLOR_LIGHT if is_light else COLOR_DARK
                
                # Highlight Last Move
                if self.board.move_stack:
                    last = self.board.peek()
                    sq_idx = self.geo_to_idx(c, r)
                    if sq_idx == last.from_square or sq_idx == last.to_square:
                        color = COLOR_HIGHLIGHT
                
                # Editor Visual: Highlight square under mouse if editing
                if self.edit_mode and self.rect().contains(self.mouse_pos):
                    mx, my = self.mouse_pos.x(), self.mouse_pos.y()
                    mc, mr = mx // sq_size, my // sq_size
                    if mc == c and mr == r:
                        color = QColor(100, 200, 255, 100) # Blue tint for editor hover

                if self.dragging and self.geo_to_idx(c, r) == self.drag_start_sq:
                    color = COLOR_SELECTED

                painter.fillRect(c*sq_size, r*sq_size, sq_size, sq_size, color)

        # 2. Engine Arrows
        if not self.dragging and not self.edit_mode:
            for move, color in self.engine_arrows:
                self.draw_arrow(painter, move, color, sq_size)

        # 3. Pieces
        for sq, piece in self.board.piece_map().items():
            if self.dragging and sq == self.drag_start_sq: continue
            c, r = self.idx_to_geo(sq)
            pm = self.piece_pixmaps.get(piece.symbol())
            if pm: painter.drawPixmap(c*sq_size, r*sq_size, pm)

        # 4. Dragged Piece (or Editor Ghost Piece)
        if self.dragging and self.drag_piece:
            pm = self.piece_pixmaps.get(self.drag_piece.symbol())
            if pm:
                x = self.mouse_pos.x() - sq_size // 2
                y = self.mouse_pos.y() - sq_size // 2
                painter.drawPixmap(x, y, pm)
        
        elif self.edit_mode and self.editor_piece:
            # Draw ghost piece attached to mouse in editor mode
            pm = self.piece_pixmaps.get(self.editor_piece.symbol())
            if pm:
                painter.setOpacity(0.7)
                x = self.mouse_pos.x() - sq_size // 2
                y = self.mouse_pos.y() - sq_size // 2
                painter.drawPixmap(x, y, pm)
                painter.setOpacity(1.0)

    def draw_arrow(self, painter, move, color, sq_size):
        c1, r1 = self.idx_to_geo(move.from_square)
        c2, r2 = self.idx_to_geo(move.to_square)
        start = QPoint(c1*sq_size + sq_size//2, r1*sq_size + sq_size//2)
        end = QPoint(c2*sq_size + sq_size//2, r2*sq_size + sq_size//2)
        pen = QPen(color)
        pen.setWidth(int(sq_size * 0.15))
        pen.setCapStyle(Qt.RoundCap)
        painter.setPen(pen)
        painter.drawLine(start, end)
        dx = end.x() - start.x()
        dy = end.y() - start.y()
        angle = math.atan2(dy, dx)
        arrow_len = sq_size * 0.35
        p1 = end
        p2 = QPoint(int(end.x() - arrow_len * math.cos(angle - math.pi/6)), int(end.y() - arrow_len * math.sin(angle - math.pi/6)))
        p3 = QPoint(int(end.x() - arrow_len * math.cos(angle + math.pi/6)), int(end.y() - arrow_len * math.sin(angle + math.pi/6)))
        painter.setBrush(QBrush(color))
        painter.setPen(Qt.NoPen)
        painter.drawPolygon(QPolygonF([QPoint(p1), QPoint(p2), QPoint(p3)]))

    def geo_to_idx(self, c, r):
        return chess.square(7-c, r) if self.flipped else chess.square(c, 7-r)

    def idx_to_geo(self, sq):
        f, r = chess.square_file(sq), chess.square_rank(sq)
        return (7-f, r) if self.flipped else (f, 7-r)

    def mousePressEvent(self, event):
        sq_size = self.width() // 8
        sq = self.geo_to_idx(event.x() // sq_size, event.y() // sq_size)
        
        # --- EDITOR MODE LOGIC ---
        if self.edit_mode:
            if event.button() == Qt.LeftButton:
                if self.editor_piece:
                    # Place Piece
                    self.board.set_piece_at(sq, self.editor_piece)
                else:
                    # Trash (Clear)
                    self.board.remove_piece_at(sq)
            elif event.button() == Qt.RightButton:
                # Right click always clears in edit mode
                self.board.remove_piece_at(sq)
            
            self.board_changed.emit()
            self.update()
            return
        # -------------------------

        # Standard Play Mode
        if event.button() == Qt.LeftButton:
            piece = self.board.piece_at(sq)
            if piece:
                self.dragging = True
                self.drag_start_sq = sq
                self.drag_piece = piece
                self.mouse_pos = event.pos()
                self.update()

    def mouseMoveEvent(self, event):
        self.mouse_pos = event.pos()
        self.update()

    def mouseReleaseEvent(self, event):
        if self.dragging:
            sq_size = self.width() // 8
            dest_sq = self.geo_to_idx(event.x() // sq_size, event.y() // sq_size)
            if dest_sq is not None and dest_sq != self.drag_start_sq:
                move = chess.Move(self.drag_start_sq, dest_sq)
                p = self.board.piece_at(self.drag_start_sq)
                if p and p.piece_type == chess.PAWN and chess.square_rank(dest_sq) in [0, 7]:
                    move.promotion = chess.QUEEN
                if move in self.board.legal_moves:
                    self.move_made.emit(move)
            self.dragging = False
            self.update()

# -------------------------
# MAIN APP (UPDATED WITH SAVER & EDITOR)
# -------------------------
class ChessAppUltra(QWidget):
    sig_start_analysis = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Chess AI Tool - Ultra Pro v2 (Trainer Edition)")
        self.setStyleSheet("""
            QWidget { background-color: #1a1a1a; color: #cfcfcf; font-family: 'Segoe UI', Arial; }
            QPushButton { padding: 8px; border-radius: 4px; font-weight: bold; }
            QTextEdit { background: #252525; border: 1px solid #444; }
        """)
        self.resize(1200, 750)
        
        self.board = chess.Board()
        self.original_img = None # Cache for training data
        
        # Setup Workers
        self.engine_thread = QThread()
        self.engine_worker = EngineWorker(STOCKFISH_PATH, STOCKFISH_DEPTH)
        self.engine_worker.moveToThread(self.engine_thread)
        self.sig_start_analysis.connect(self.engine_worker.analyze)
        self.engine_worker.result_ready.connect(self.on_engine_result)
        self.engine_thread.start()
        
        self.init_ui()

    def init_ui(self):
        main = QHBoxLayout()
        
        # --- LEFT: Board Area ---
        board_area = QHBoxLayout()
        self.eval_widget = EvalBarWidget()
        
        # Container for palette (initially hidden) and board
        self.board_container = QHBoxLayout()
        self.palette = PiecePalette()
        self.palette.hide() # Hidden until edit mode
        self.palette.piece_selected.connect(self.on_palette_select)
        
        self.board_widget = ChessBoardWidget(self.board)
        self.board_widget.move_made.connect(self.user_move)
        self.board_widget.board_changed.connect(self.update_board_state) # For editor updates
        
        board_area.addWidget(self.eval_widget)
        board_area.addWidget(self.palette)
        board_area.addWidget(self.board_widget)
        
        # --- RIGHT: Controls ---
        controls = QVBoxLayout()
        controls.setContentsMargins(20, 0, 20, 0)
        
        lbl_title = QLabel("ANALYSIS & TRAINING")
        lbl_title.setStyleSheet("font-size: 18px; color: #769656; font-weight: bold;")
        self.lbl_status = QLabel("Ready")
        
        # Analysis Box
        self.txt_analysis = QTextEdit()
        self.txt_analysis.setPlaceholderText("Engine lines...")
        self.txt_analysis.setReadOnly(True)
        self.txt_analysis.setMaximumHeight(150)
        
        # Turn
        turn_box = QFrame()
        turn_box.setStyleSheet("background: #222; border-radius: 4px;")
        turn_layout = QHBoxLayout(turn_box)
        self.rb_white = QRadioButton("White")
        self.rb_white.setChecked(True)
        self.rb_black = QRadioButton("Black")
        turn_layout.addWidget(QLabel("To Move:"))
        turn_layout.addWidget(self.rb_white)
        turn_layout.addWidget(self.rb_black)
        self.rb_white.clicked.connect(self.sync_turn)
        self.rb_black.clicked.connect(self.sync_turn)

        # Buttons
        btn_load = QPushButton("📂 Load Image")
        btn_load.clicked.connect(self.load_image)
        btn_load.setStyleSheet("background: #007acc; color: white;")
        
        self.btn_play_best = QPushButton("⚡ Play Best Move")
        self.btn_play_best.clicked.connect(self.play_engine_best)
        self.btn_play_best.setStyleSheet("background: #d48806; color: white;")

        # --- EDIT & TRAIN CONTROLS ---
        edit_layout = QHBoxLayout()
        self.btn_edit = QPushButton("✏️ Edit Board")
        self.btn_edit.setCheckable(True)
        self.btn_edit.clicked.connect(self.toggle_edit_mode)
        self.btn_edit.setStyleSheet("background: #444; border: 1px solid #666;")
        
        self.btn_save_train = QPushButton("💾 Save Correction")
        self.btn_save_train.clicked.connect(self.save_training_data)
        self.btn_save_train.setStyleSheet("background: #882222; color: white;")
        self.btn_save_train.setToolTip("Save Image + Corrected FEN for retraining")
        
        edit_layout.addWidget(self.btn_edit)
        edit_layout.addWidget(self.btn_save_train)

        # Basic Controls
        row_btns = QHBoxLayout()
        btn_undo = QPushButton("Undo")
        btn_undo.clicked.connect(self.undo_move)
        btn_flip = QPushButton("Flip")
        btn_flip.clicked.connect(self.flip_board)
        row_btns.addWidget(btn_undo)
        row_btns.addWidget(btn_flip)
        
        self.btn_engine_toggle = QPushButton("Enable Engine")
        self.btn_engine_toggle.setCheckable(True)
        self.btn_engine_toggle.clicked.connect(self.toggle_engine)

        # Assemble Right Panel
        controls.addWidget(lbl_title)
        controls.addWidget(self.lbl_status)
        controls.addSpacing(10)
        controls.addWidget(turn_box)
        controls.addWidget(self.txt_analysis)
        controls.addSpacing(10)
        controls.addWidget(self.btn_play_best)
        controls.addWidget(btn_load)
        controls.addSpacing(10)
        controls.addLayout(edit_layout) # New Edit Section
        controls.addSpacing(10)
        controls.addLayout(row_btns)
        controls.addWidget(self.btn_engine_toggle)
        controls.addStretch()

        main.addLayout(board_area)
        main.addLayout(controls)
        self.setLayout(main)

    # --- ACTIONS ---
    def user_move(self, move):
        self.board.push(move)
        self.update_board_state()

    def sync_turn(self):
        self.board.turn = chess.WHITE if self.rb_white.isChecked() else chess.BLACK
        self.update_board_state()

    def undo_move(self):
        if self.board.move_stack:
            self.board.pop()
            self.update_board_state()

    def flip_board(self):
        self.board_widget.flipped = not self.board_widget.flipped
        self.board_widget.update()

    def update_board_state(self):
        self.board_widget.engine_arrows = [] 
        self.board_widget.update()
        if self.btn_engine_toggle.isChecked():
            self.lbl_status.setText("Analyzing...")
            self.sig_start_analysis.emit(self.board.fen())

    def toggle_engine(self):
        if self.btn_engine_toggle.isChecked():
            self.sig_start_analysis.emit(self.board.fen())
        else:
            self.board_widget.engine_arrows = []
            self.board_widget.update()
            self.eval_widget.set_eval(0)

    def on_engine_result(self, top_moves, val, is_mate):
        self.lbl_status.setText("Analysis Complete")
        self.eval_widget.set_eval(val, is_mate)
        self.board_widget.set_arrows(top_moves)
        txt = ""
        for i, m in enumerate(top_moves):
            uci = m.get('Move')
            cp = m.get('Centipawn')
            mate = m.get('Mate')
            score = f"Mate {mate}" if mate else f"{cp/100:+.2f}"
            txt += f"{i+1}. {uci} ({score})\n"
        self.txt_analysis.setText(txt)

    def play_engine_best(self):
        if self.board_widget.engine_arrows:
            best_move = self.board_widget.engine_arrows[0][0]
            self.board.push(best_move)
            self.update_board_state()
        else:
            if not self.btn_engine_toggle.isChecked():
                self.btn_engine_toggle.setChecked(True)
                self.update_board_state()
            QMessageBox.information(self, "Info", "Wait for analysis...")

    # --- IMAGE & TRAINING LOGIC ---
    def load_image(self):
        f, _ = QFileDialog.getOpenFileName(self, "Load Image")
        if not f: return
        self.lbl_status.setText("Processing Vision...")
        img = cv2.imread(f)
        self.original_img = img # Cache for training save
        
        img_crop = auto_crop_board(img)
        if img_crop is None: img_crop = img
        
        self.yolo_worker = YoloWorker(MODEL_PATH, img_crop)
        self.yolo_worker.result_ready.connect(self.on_image_loaded)
        self.yolo_worker.start()

    def on_image_loaded(self, fen):
        if fen:
            self.board = chess.Board(fen)
            # Auto-detect turn? For now default to UI
            self.board.turn = chess.WHITE if self.rb_white.isChecked() else chess.BLACK
            self.board_widget.board = self.board
            self.update_board_state()
            self.lbl_status.setText("Board Loaded")
        else:
            self.lbl_status.setText("Vision Failed")

    def toggle_edit_mode(self):
        is_editing = self.btn_edit.isChecked()
        self.board_widget.edit_mode = is_editing
        
        if is_editing:
            self.palette.show()
            self.lbl_status.setText("Editing Mode ON")
            self.btn_edit.setStyleSheet("background: #00AA00; color: white;")
        else:
            self.palette.hide()
            self.lbl_status.setText("Ready")
            self.btn_edit.setStyleSheet("background: #444; border: 1px solid #666;")

    def on_palette_select(self, piece):
        self.board_widget.editor_piece = piece

    def save_training_data(self):
        # 1. Check if we have an image
        if self.original_img is None:
            QMessageBox.warning(self, "Error", "No image loaded to save!")
            return

        # 2. Setup Directory
        save_dir = "yolo_retrain_data"
        os.makedirs(save_dir, exist_ok=True)
        
        ts = int(time.time())
        img_name = f"chess_{ts}.jpg"
        lbl_name = f"chess_{ts}.txt"
        
        # --- CRITICAL FIX: Warp the board before saving ---
        # The auto-labeling script assumes the image is a perfect 8x8 grid.
        # So we must run the warp detection before saving.
        try:
            warped_img = detect_and_warp_board(self.original_img)
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Could not warp board for saving: {e}")
            return
        
        # 3. Save the WARPED Image
        cv2.imwrite(os.path.join(save_dir, img_name), warped_img)
        
        # 4. Save the Corrected FEN (Label)
        with open(os.path.join(save_dir, lbl_name), "w") as f:
            f.write(self.board.fen())
            
        QMessageBox.information(self, "Success", 
            f"Saved Training Pair!\nImage: {img_name}\nLabel: Auto-generated from FEN")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = ChessAppUltra()
    w.show()
    sys.exit(app.exec_())
