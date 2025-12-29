import cv2
import numpy as np
import subprocess
import string
import json
import copy
import time
import signal
import sys

# --------------------------
# CONFIG
# --------------------------
SCREENSHOT_PATH = "screen.png"
DEBUG_IMAGE_PATH = "debug_bottles.png"
BOTTLE_DATA_FILE = "bottle_data.json"
BOTTLE_COORDS_PATH = "bottle_coords.json"
TAP_LOG_PATH = "tap_log.json"

# Detection params
TOP_PCT = 0.15
BOTTOM_PCT = 0.74  # lowered to avoid detecting UI/phantom bottle at bottom
MIN_BOTTLE_WIDTH = 40
MIN_BOTTLE_HEIGHT = 150
COLOR_DIFF_THRESHOLD = 40
EMPTY_PIXEL_RATIO = 0.9
SEG_EMPTY_PIXEL_RATIO = 0.82  # per-segment empty/background ratio threshold
ROW_THRESHOLD = 50
TOP_MARGIN_SEGMENTS = 0.20

# Solver params
MAX_HEIGHT = 4
MAX_DEPTH = 500
MAX_ITERATIONS = 50

# Execution params
MOVE_DELAY = 2.0
TAP_STEP_DELAY = 0.15

# Global state
tap_log = []
all_moves_executed = []
bottle_contents = {}  # Track what color is in each bottle: {bottle_num: color}
last_pour_dst = None
last_pour_color = None





# Learning / anti-loop memory: avoid repeating the same board situation
seen_state_signatures = set()


# Persistence for learned reveals across runs
MEMORY_FILE = "water_sort_memory.json"

# Deadlock thresholds (tune as needed)
DEADLOCK_REPEAT_LIMIT = 6          # same board seen this many times
DEADLOCK_NOINFO_LIMIT = 12         # iterations without reducing '?' count

_state_seen_counts = {}
_prev_unknowns = None
_noinfo_streak = 0


def save_learning_snapshot(reason=""):
    """Save current logical knowledge (revealed slots + color registry seed) to disk."""
    try:
        payload = {
            "version": 1,
            "timestamp": time.time(),
            "reason": reason,
            "bottle_count": len(state_board) if state_board is not None else 0,
            "state_board": state_board,
            "registry_seed_lab": color_registry.export_seed_lab(),
        }
        with open(MEMORY_FILE, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"\n[MEMORY] Saved learning snapshot to {MEMORY_FILE} ({reason})")
    except Exception as e:
        print(f"\n[MEMORY] Failed to save snapshot: {e}")


def load_learning_snapshot(current_bottle_count):
    """Load prior knowledge if compatible with current puzzle size."""
    try:
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if payload.get("bottle_count") != current_bottle_count:
            return None
        seed = payload.get("registry_seed_lab")
        if isinstance(seed, dict) and seed:
            color_registry.load_seed_lab(seed)
        return payload
    except FileNotFoundError:
        return None
    except Exception as e:
        print(f"[MEMORY] Could not load snapshot: {e}")
        return None


def apply_snapshot_to_state(snapshot):
    """Merge saved revealed info into current state_board (only fills '?' where snapshot has known)."""
    global state_board
    if not snapshot or state_board is None:
        return
    saved = snapshot.get("state_board")
    if not saved or len(saved) != len(state_board):
        return
    for i in range(len(state_board)):
        cur = list((state_board[i] + ["BC"] * 4)[:4])
        prev = list((saved[i] + ["BC"] * 4)[:4])
        for k in range(4):
            if cur[k] == "?" and prev[k] not in ("?", "BC"):
                cur[k] = prev[k]
        state_board[i] = cur


def board_signature(board):
    """Hashable signature of current logical board (4 fixed slots per bottle)."""
    return tuple(tuple((b + ["BC"] * 4)[:4]) for b in board)

def count_unknowns(board):
    return sum(1 for b in board for s in (b + ["BC"] * 4)[:4] if s == "?")

def save_memory_snapshot(path="water_sort_memory.json"):
    """Save revealed state_board and color registry seeds to disk."""
    data = {
        "state_board": state_board,
        "bottle_contents": bottle_contents,
    }
    # Try to persist color registry if present
    try:
        if "color_registry" in globals():
            data["color_registry_seed_lab"] = color_registry.export_seed_lab()
    except Exception:
        pass

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def simulate_move_on_board(board, src_idx, dst_idx, tracked_top_color=None):
    """Simulate a move on a COPY of board using 4-slot fixed logic.

    src_idx/dst_idx are 1-based. tracked_top_color can be provided if src top is '?'.
    Returns new_board or None if move invalid.
    """
    new_board = [list((b + ["BC"] * 4)[:4]) for b in board]
    s = src_idx - 1
    d = dst_idx - 1
    src = new_board[s]
    dst = new_board[d]

    # find topmost filled in src
    try:
        top_i = next(i for i, v in enumerate(src) if v != "BC")
    except StopIteration:
        return None

    color = src[top_i]
    if color == "?":
        if not tracked_top_color:
            return None
        color = tracked_top_color

    # destination compatibility
    dst_top = None
    for v in dst:
        if v != "BC":
            dst_top = v
        break
    if dst_top is not None and dst_top not in ("?", color):
        return None

    # count contiguous color starting at top_i (treat '?' as same if tracked_top_color matches)
    cnt = 0
    for j in range(top_i, 4):
        v = src[j]
        if v == color or (v == "?" and tracked_top_color == color):
            cnt += 1
        else:
            break

    empties = [i for i, v in enumerate(dst) if v == "BC"]
    amt = min(cnt, len(empties))
    if amt <= 0:
        return None

    # empty poured slots in src
    for k in range(amt):
        src[top_i + k] = "BC"

    # fill destination bottom-up
    for _ in range(amt):
        empties = [i for i, v in enumerate(dst) if v == "BC"]
        if not empties:
            break
        dst[max(empties)] = color

    new_board[s] = src
    new_board[d] = dst
    return new_board
# Persistent logical board state (top at index 0, includes 'BC' empties and '?' unknowns)
state_board = None  # type: list[list[str]] | None

def init_state_board_from_detection(bottle_data, background_color=None):
    """Initialize persistent board from detection, forcing empty-glass segments to 'BC'.

    background_color is optional; if provided, we will re-check each segment using is_empty_segment
    so that empty glass never initializes as a real color.
    """
    global state_board
    state_board = []
    for b in bottle_data:
        segs = list(b.get('segments', []))
        segs = (segs + ["BC"] * 4)[:4]

        if background_color is not None:
            # Re-evaluate empties from the actual segment images if available
            # bottle_data here doesn't include images, so we rely on labels except we will
            # later correct via merge_detection_into_state (which keeps BC once set).
            pass

        state_board.append(segs)

def merge_detection_into_state(bottle_data):
    """Merge current detection into logical state.

    Rules:
    - state_board is the source of truth.
    - Detection may only reveal '?' slots (and never overwrite known colors).
    - If bottle count changes (ghost bottle appears/disappears), reinitialize safely to avoid crashes.
    """
    global state_board, seen_state_signatures, bottle_contents, last_pour_dst

    # Initialize or resync if bottle count differs
    if state_board is None or len(state_board) != len(bottle_data):
        # Hard resync: rebuild state_board from detection (4 slots per bottle)
        state_board = []
        for b in bottle_data:
            segs = list(b.get("segments", []))
            segs = (segs + ["BC"] * 4)[:4]
            state_board.append(segs)

        # Clear move-memory because the topology changed
        seen_state_signatures = set()
        # Tracking becomes unreliable across topology change; clear it
        bottle_contents = {}
        last_pour_dst = None
        return

    # Normal merge: reveal only
    for i, b in enumerate(bottle_data):
        det = (list(b.get("segments", [])) + ["BC"] * 4)[:4]
        st = (list(state_board[i]) + ["BC"] * 4)[:4]

        merged = []
        for s_old, s_det in zip(st, det):
            if s_old == "BC":
                merged.append("BC")
            elif s_old == "?":
                merged.append(s_det if s_det not in ("BC", "?") else "?")
            else:
                merged.append(s_old)

        state_board[i] = merged

def apply_move_to_state(from_idx, to_idx):
    """Apply a pour move to the persistent state_board (1-based bottle numbers).

    Representation: exactly 4 fixed slots TOP->BOTTOM.
      - 'BC' means empty glass.
      - '?' means hidden liquid.
      - 'A'.. means known liquid.

    Slots do NOT shift. Pouring empties the *topmost filled* slots.
    Filling a destination occupies the lowest available 'BC' slots (bottom-up).

    If the source topmost filled is '?', we use bottle_contents[from_idx] if available.
    """
    global state_board, last_pour_dst, last_pour_color, bottle_contents
    if state_board is None:
        return

    fi = from_idx - 1
    ti = to_idx - 1
    if fi < 0 or ti < 0 or fi >= len(state_board) or ti >= len(state_board):
        return

    src = (list(state_board[fi]) + ["BC"] * 4)[:4]
    dst = (list(state_board[ti]) + ["BC"] * 4)[:4]

    # Find topmost filled slot in source (first non-BC)
    try:
        src_top_i = next(i for i, s in enumerate(src) if s != "BC")
    except StopIteration:
        return

    src_top_val = src[src_top_i]
    if src_top_val == "?":
        tracked = bottle_contents.get(from_idx)
        if not tracked:
            return
        color = tracked
    else:
        color = src_top_val

    # Destination compatibility: its topmost filled (first non-BC) must be same color or unknown
    dst_top_val = None
    for s in dst:
        if s != "BC":
            dst_top_val = s
            break
    if dst_top_val is not None and dst_top_val not in ("?", color):
        return

    # Count contiguous same-color starting at src_top_i
    count = 0
    for j in range(src_top_i, 4):
        v = src[j]
        if v == color or (v == "?" and bottle_contents.get(from_idx) == color):
            count += 1
        else:
            break

    empty_idxs = [i for i, s in enumerate(dst) if s == "BC"]
    pour_amt = min(count, len(empty_idxs))
    if pour_amt <= 0:
        return

    # Empty the poured slots in source (no shifting)
    for k in range(pour_amt):
        src[src_top_i + k] = "BC"

    # Fill destination from bottom-most empties upward
    for _ in range(pour_amt):
        empties = [i for i, s in enumerate(dst) if s == "BC"]
        if not empties:
            break
        dst[max(empties)] = color

    state_board[fi] = src
    state_board[ti] = dst

    last_pour_dst = to_idx
    last_pour_color = color
    bottle_contents[to_idx] = color
# --------------------------
# PERSISTENT COLOR REGISTRY (stable IDs across scans)
# --------------------------
class ColorRegistry:
    """Assign stable labels to colors across iterations by matching in Lab space.

    Labels are used only as internal IDs for the solver (A, B, C...).
    """

    def __init__(self, dist_threshold=18.0):
        self.dist_threshold = float(dist_threshold)
        self._colors_lab = []  # list[np.ndarray shape(3,)]
        self._labels = []      # list[str]

    @staticmethod
    def bgr_to_lab(bgr_tuple):
        bgr = np.uint8([[list(bgr_tuple)]])  # shape (1,1,3)
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)[0, 0].astype(np.float32)
        return lab

    @staticmethod
    def _label_from_index(idx: int) -> str:
        """0->A, 1->B, ... 25->Z, 26->AA, etc."""
        letters = string.ascii_uppercase
        out = ""
        n = idx
        while True:
            out = letters[n % 26] + out
            n = n // 26 - 1
            if n < 0:
                break
        return out

    def match_or_create(self, lab_color):
        """Return existing label if close enough; otherwise create a new label."""
        lab = np.array(lab_color, dtype=np.float32).reshape(3,)
        if not self._colors_lab:
            label = self._label_from_index(0)
            self._colors_lab.append(lab)
            self._labels.append(label)
            return label

        dists = [float(np.linalg.norm(lab - c)) for c in self._colors_lab]
        best_i = int(np.argmin(dists))
        if dists[best_i] <= self.dist_threshold:
            return self._labels[best_i]

        label = self._label_from_index(len(self._labels))
        self._colors_lab.append(lab)
        self._labels.append(label)
        return label

    # ---- Persistence helpers ----
    def export_seed_lab(self):
        """Export registry to JSON-serializable structure."""
        return [
            {"label": lab, "lab": [float(x) for x in col.tolist()]}
            for lab, col in zip(self._labels, self._colors_lab)
        ]

    def load_seed_lab(self, items):
        """Load registry from export_seed_lab output."""
        self._labels = []
        self._colors_lab = []
        if not items:
            return
        for it in items:
            lab = it.get("lab")
            label = it.get("label")
            if label is None or lab is None:
                continue
            self._labels.append(str(label))
            self._colors_lab.append(np.array(lab, dtype=np.float32).reshape(3,))


def load_seed_lab(self, label_to_lab):
    """Seed registry with previously known labels -> LAB triples."""
    self._colors_lab = []
    self._labels = []
    for label in sorted(label_to_lab.keys(), key=lambda x: (len(x), x)):
        lab = np.array(label_to_lab[label], dtype=np.float32)
        self._colors_lab.append(lab)
        self._labels.append(label)

def export_seed_lab(self):
    """Export registry labels and LAB centers for persistence."""
    return {label: [float(x) for x in lab.tolist()] for label, lab in zip(self._labels, self._colors_lab)}

    def match_or_create(self, bgr_tuple):
        lab = self.bgr_to_lab(bgr_tuple)

        if not self._colors_lab:
            label = "A"
            self._colors_lab.append(lab)
            self._labels.append(label)
            return label

        # Nearest-neighbor match
        dists = [float(np.linalg.norm(c - lab)) for c in self._colors_lab]
        best_i = int(np.argmin(dists))
        if dists[best_i] <= self.dist_threshold:
            return self._labels[best_i]

        # Create new stable label
        next_idx = len(self._labels)
        if next_idx < 26:
            label = chr(ord('A') + next_idx)
        else:
            label = f"C{next_idx}"
        self._colors_lab.append(lab)
        self._labels.append(label)
        return label

# Global persistent registry (persists across detect_and_analyze calls)
color_registry = ColorRegistry(dist_threshold=18.0)

def is_unknown_segment(seg_bgr, background_color):
    """Heuristic detection for mystery '?' segments (dark fill with white glyph)."""
    pixels = seg_bgr.reshape(-1, 3).astype(np.int16)
    # White pixels (question mark glyph)
    white = np.mean((pixels[:,0] > 220) & (pixels[:,1] > 220) & (pixels[:,2] > 220))
    # Dark pixels (mystery fill tends to be very dark)
    brightness = pixels.mean(axis=1)
    dark = np.mean(brightness < 70)
    # Not background (avoid treating empty glass highlights as unknown)
    bg = np.array(background_color, dtype=np.int16)
    dist_bg = np.linalg.norm(pixels - bg, axis=1)
    not_bg = np.mean(dist_bg > COLOR_DIFF_THRESHOLD)

    return (white > 0.006) and (dark > 0.35) and (not_bg > 0.25)


def is_empty_segment(seg_bgr, background_color):
    """Robust per-segment empty detection.

    We treat a segment as empty glass if:
      - its center is mostly background-colored (high bg_ratio), and
      - it has little saturation / little darkness (typical of background + highlights)

    Returns (is_empty: bool, core_mask: np.ndarray[bool], center_crop: np.ndarray).
    """
    h, w = seg_bgr.shape[:2]
    x0 = int(w * 0.18); x1 = int(w * 0.82)
    y0 = int(h * 0.12); y1 = int(h * 0.88)
    center = seg_bgr[y0:y1, x0:x1]

    pixels = center.reshape(-1, 3).astype(np.int16)
    bg = np.array(background_color, dtype=np.int16)

    dist_bg = np.linalg.norm(pixels - bg, axis=1)

    # Background ratio on center crop
    bg_ratio = float((dist_bg < COLOR_DIFF_THRESHOLD).mean())

    # Core pixels = not background AND not near-white highlight
    non_bg = dist_bg > COLOR_DIFF_THRESHOLD
    white = (pixels[:,0] > 230) & (pixels[:,1] > 230) & (pixels[:,2] > 230)
    core = non_bg & (~white)
    core_ratio = float(core.mean())

    hsv = cv2.cvtColor(center, cv2.COLOR_BGR2HSV)
    sat = hsv[:,:,1].astype(np.float32)
    val = hsv[:,:,2].astype(np.float32)

    sat_mean = float(sat.mean())
    sat_p90  = float(np.percentile(sat, 90))
    dark_ratio = float((val < 90).mean())
    v_p10 = float(np.percentile(val, 10))

    # Absolute empty: mostly background in the interior
    # This is the key rule that prevents "empty space becomes G".
    if bg_ratio > 0.86 and sat_mean < 70 and dark_ratio < 0.06:
        return True, core, center

    # Strong empty signal: almost no core pixels
    if core_ratio < 0.08:
        return True, core, center

    # Empty glass: low saturation and almost no dark pixels and weak core
    if sat_mean < 38 and sat_p90 < 70 and dark_ratio < 0.02 and core_ratio < 0.20:
        return True, core, center

    # Very bright interior is typically empty (background showing through)
    if v_p10 > 175 and sat_mean < 60 and core_ratio < 0.20:
        return True, core, center

    return False, core, center
# --------------------------
# SIGNAL HANDLER
# --------------------------
def signal_handler(sig, frame):
    print("\nFailsafe activated. Stopping execution.")
    with open(TAP_LOG_PATH, "w") as f:
        json.dump(tap_log, f, indent=2)
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

# --------------------------
# DETECTION FUNCTIONS
# --------------------------
def take_screenshot():
    """Capture a device screenshot to SCREENSHOT_PATH.

    ADB screencap can intermittently fail on some devices/USB states (return code 0xFFFFFFFF).
    We retry a few times and fall back to: adb shell screencap -> adb pull.
    """
    screenshot_path = SCREENSHOT_PATH

    def _run(cmd, **kwargs):
        return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, **kwargs)

    def _adb_devices():
        p = _run(["adb", "devices"])
        out = (p.stdout or b"").decode(errors="ignore")
        return out

    def _try_exec_out():
        with open(screenshot_path, "wb") as f:
            subprocess.run(
                ["adb", "exec-out", "screencap", "-p"],
                stdout=f,
                stderr=subprocess.DEVNULL,
                check=True
            )
        return True

    def _try_shell_pull():
        remote = "/sdcard/__clause_tmp_screen.png"
        # capture on device
        subprocess.run(["adb", "shell", "screencap", "-p", remote], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        # pull to local
        subprocess.run(["adb", "pull", remote, screenshot_path], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        # cleanup best-effort
        subprocess.run(["adb", "shell", "rm", "-f", remote],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True

    last_err = None
    for attempt in range(1, 5):
        try:
            _try_exec_out()
            return screenshot_path
        except Exception as e:
            last_err = e
            # Try to recover the ADB connection
            try:
                subprocess.run(["adb", "wait-for-device"], timeout=5,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except Exception:
                pass
            # Some setups benefit from reconnect
            subprocess.run(["adb", "reconnect"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            time.sleep(0.25 * attempt)

    # Fallback method
    try:
        _try_shell_pull()
        return screenshot_path
    except Exception as e2:
        # Give a helpful error with device status
        devices = _adb_devices()
        msg = (
            "\n\nADB screenshot capture failed.\n"
            "Tried: adb exec-out screencap -p (with retries) and adb shell screencap + pull.\n"
            "Most common causes: no device connected, device unauthorized, USB debugging off, or ADB server issue.\n"
            f"adb devices output:\n{devices}\n"
            f"Last exec-out error: {repr(last_err)}\n"
            f"Fallback error: {repr(e2)}\n"
        )
        raise RuntimeError(msg)

def get_debug_crop(img):
    h, w = img.shape[:2]
    y_min = int(h * TOP_PCT)
    y_max = int(h * BOTTOM_PCT)
    return img[y_min:y_max, :], y_min

def detect_bottles(img):
    h, w = img.shape[:2]
    y_max = int(h * BOTTOM_PCT)
    cropped = img[0:y_max, :]
    
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blur, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bottle_rects = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w >= MIN_BOTTLE_WIDTH and h >= MIN_BOTTLE_HEIGHT:
            bottle_rects.append((x, y, w, h))
    return bottle_rects

def detect_background(img, bottle_rects):
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    for x, y, w, h in bottle_rects:
        cv2.rectangle(mask, (x, y), (x+w, y+h), 255, -1)
    bg_pixels = img[mask == 0]
    return tuple(map(int, np.median(bg_pixels, axis=0)))

def group_bottles_by_row(bottle_rects):
    rows = []
    for rect in bottle_rects:
        x, y, w, h = rect
        placed = False
        for row in rows:
            if abs(y - row[0][1]) <= ROW_THRESHOLD:
                row.append(rect)
                placed = True
                break
        if not placed:
            rows.append([rect])
    rows.sort(key=lambda row: min(y for (x,y,w,h) in row))
    for row in rows:
        row.sort(key=lambda r: r[0])
    return rows


def analyze_bottles(img, bottle_rects, background_color, force_last_two_empty=True):
    """Detect bottles and assign stable color labels across scans.

    Returns: list of {bottle: int, segments: [label|'?'|'BC', ...]} where segments are top->bottom.
    """
    bottle_data = []
    rows = group_bottles_by_row(bottle_rects)

    all_bottles = [b for row in rows for b in row]
    total_bottles = len(all_bottles)

    for idx, (x, y, w, h) in enumerate(all_bottles):
        bottle_num = idx + 1
        entry = {"bottle": bottle_num, "segments": []}

        roi = img[y:y+h, x:x+w]
        dist = np.linalg.norm(roi.reshape(-1,3) - np.array(background_color), axis=1)
        ratio_bg = float(np.mean(dist < COLOR_DIFF_THRESHOLD))
        is_empty_bottle = ratio_bg >= EMPTY_PIXEL_RATIO

        force_empty = (force_last_two_empty and idx >= total_bottles - 2)

        # Debug rectangle
        cv2.rectangle(img, (x, y), (x+w, y+h), (0,255,0), 2)

        y_adj = y + int(h * TOP_MARGIN_SEGMENTS)
        adj_h = h - int(h * TOP_MARGIN_SEGMENTS)
        layer_h = max(1, adj_h // 4)

        for i in range(4):
            ly = y_adj + i*layer_h
            seg = img[ly:ly+layer_h, x:x+w]
            pixels = seg.reshape(-1,3)

            dist_seg = np.linalg.norm(pixels.astype(np.int16) - np.array(background_color, dtype=np.int16), axis=1)
            ratio_bg_seg = float(np.mean(dist_seg < COLOR_DIFF_THRESHOLD))

            # Segment-level empty detection (robust to glass highlights)
            is_empty_seg, core_mask, center_crop = is_empty_segment(seg, background_color)

            if is_empty_bottle or ratio_bg_seg >= SEG_EMPTY_PIXEL_RATIO or is_empty_seg or np.sum(core_mask) == 0 or force_empty:
                seg_id = "BC"
            else:
                # Heuristic: mystery '?' segments
                if is_unknown_segment(seg, background_color):
                    seg_id = "?"
                else:
                    med = tuple(np.median(center_crop.reshape(-1,3)[core_mask], axis=0).astype(int))
                    seg_id = color_registry.match_or_create(med)

            entry["segments"].append(seg_id)
            cv2.putText(img, str(seg_id), (x+2, ly+16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)

        bottle_data.append(entry)

    return bottle_data


def detect_and_analyze(force_last_two_empty=True):
    """Take screenshot and analyze bottles"""
    take_screenshot()
    img_full = cv2.imread(SCREENSHOT_PATH)
    if img_full is None:
        raise RuntimeError("Failed to load screenshot")

    bottle_rects = detect_bottles(img_full)
    background_color = detect_background(img_full, bottle_rects)
    bottle_data = analyze_bottles(img_full, bottle_rects, background_color, force_last_two_empty=force_last_two_empty)

    # Update persistent logical state (prevents empty glass becoming a color later)
    merge_detection_into_state(bottle_data)
    if state_board is not None:
        seen_state_signatures.add(board_signature(state_board))

    debug_img, _ = get_debug_crop(img_full.copy())

    # Sort bottle_rects the same way as analyze_bottles does
    rows = group_bottles_by_row(bottle_rects)
    all_bottles = [b for row in rows for b in row]

    bottle_coords = {}
    for idx, rect in enumerate(all_bottles):
        tap_x = int(rect[0] + rect[2]/2)
        tap_y = int(rect[1] + rect[3]/2)
        bottle_num = idx + 1
        bottle_coords[str(bottle_num)] = [tap_x, tap_y]

        debug_x = tap_x
        debug_y = tap_y - int(img_full.shape[0]*TOP_PCT)
        cv2.circle(debug_img, (debug_x, debug_y), 5, (0,0,255), -1)
        cv2.putText(debug_img, str(bottle_num), (debug_x-10, debug_y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    cv2.imwrite(DEBUG_IMAGE_PATH, debug_img)

    with open(BOTTLE_DATA_FILE, "w") as f:
        json.dump(bottle_data, f, indent=2)
    with open(BOTTLE_COORDS_PATH, "w") as f:
        json.dump(bottle_coords, f, indent=2)

    print(f"Detection: {len(bottle_data)} bottles")

    return bottle_data, bottle_coords

# --------------------------
# SOLVER FUNCTIONS
# --------------------------
def convert_to_solver_format(bottle_data):
    """
    Convert logical state to solver format using TRUE water-sort rules.

    Tracking rules (IMPORTANT):
    - Bottles are NOT tracked by default
    - Track ONLY if:
        1) Bottle was empty and received a pour
        2) Bottle contains exactly ONE known color and no '?'
    - Bottles with '?' underneath are NEVER locked/tracked
    """
    global bottle_contents, state_board, last_pour_dst

    bottles = []

    for i, b in enumerate(bottle_data):
        bottle_num = i + 1
        st = (list(state_board[i]) + ["BC"] * 4)[:4]

        # Solver bottle = strip BC
        segments = [x for x in st if x != "BC"]
        bottles.append(segments)

        # ---- Tracking logic ----
        non_empty = [x for x in st if x != "BC"]

        # Case 1: Truly empty bottle → clear tracking
        if not non_empty:
            if bottle_num in bottle_contents:
                del bottle_contents[bottle_num]
            continue

        # Case 2: Bottle just received a pour into empty
        if bottle_num == last_pour_dst:
            bottle_contents[bottle_num] = non_empty[-1]
            continue

        # Case 3: Bottle contains exactly one known color, no '?'
        unique_colors = set(non_empty)
        if "?" not in unique_colors and len(unique_colors) == 1:
            bottle_contents[bottle_num] = next(iter(unique_colors))
        else:
            # Mixed or unknown → do NOT track
            if bottle_num in bottle_contents:
                del bottle_contents[bottle_num]

    return bottles


def is_uniform_full(bottle):
    if '?' in bottle:
        return False
    return len(bottle) == MAX_HEIGHT and all(s == bottle[0] for s in bottle)

def is_almost_full_same(bottle):
    if '?' in bottle or len(bottle) != MAX_HEIGHT - 1:
        return False
    return all(s == bottle[0] for s in bottle)

def is_solved(board):
    for b in board:
        if '?' in b or (b and not is_uniform_full(b)):
            return False
    return True

def has_unknowns(board):
    return any('?' in b for b in board)

def can_pour(from_b, to_b, from_idx, to_idx):
    """Check if we can pour from_b into to_b, using logical tracking"""
    global bottle_contents
    
    # Can't pour from empty or if top is unknown
    if not from_b or '?' in from_b[0:1]:
        return False
    
    # Don't pour from completed bottles
    if is_uniform_full(from_b) or is_almost_full_same(from_b):
        return False
    
    # Can't pour if destination is full
    if len(to_b) >= MAX_HEIGHT:
        return False
    
    # Get the color we're trying to pour
    color_to_pour = from_b[0]
    
    # Check if destination bottle has a tracked color
    to_bottle_num = to_idx + 1
    if to_bottle_num in bottle_contents:
        # Bottle has a color already, must match
        return color_to_pour == bottle_contents[to_bottle_num]
    
    # If no tracking and bottle appears empty, can pour
    if not to_b:
        return True
    
    # Can't pour onto unknown
    if '?' in to_b[0:1]:
        return False
    
    # Can only pour if colors match
    return from_b[0] == to_b[0]

def do_pour(board, i, j):
    """Pour from bottle i to bottle j"""
    global bottle_contents
    
    fb = board[i]
    tb = board[j]
    if not can_pour(fb, tb, i, j):
        return 0
    color = fb[0]
    
    # Track what color goes into destination
    to_bottle_num = j + 1
    if to_bottle_num not in bottle_contents:
        bottle_contents[to_bottle_num] = color
        print(f"      Tracking: Bottle {to_bottle_num} now contains {color}")
    
    moved = 0
    while fb and fb[0] == color and len(tb) < MAX_HEIGHT:
        tb.insert(0, fb.pop(0))
        moved += 1
    
    # If source is now empty, clear its tracking
    if not fb:
        from_bottle_num = i + 1
        if from_bottle_num in bottle_contents:
            del bottle_contents[from_bottle_num]
            print(f"      Tracking: Bottle {from_bottle_num} is now empty")
    
    return moved

def serialize_board(board):
    return tuple(tuple(b) for b in board)

def score_strategic_move(from_bottle, to_bottle):
    """
    Score a move for revealing unknowns while avoiding pointless shuffling.
    """
    score = 0

    # High priority: moving reveals unknown underneath
    if len(from_bottle) > 1 and '?' in from_bottle[1:]:
        score += 100

    # Pour into empty bottle
    if not to_bottle:
        # Valuable only if it reveals information
        if len(from_bottle) > 1 and '?' in from_bottle[1:]:
            score += 80
        elif '?' in from_bottle:
            score += 25
        else:
            score += 5

        # Penalize pointless singleton shuffles (e.g., grey 10 -> 11)
        if len(from_bottle) == 1 and '?' not in from_bottle:
            score -= 10

    # Matching colors (normal water sort behavior)
    if to_bottle and from_bottle[0] == to_bottle[0]:
        score += 40

    # Moving from mixed bottle
    if len(set(from_bottle)) > 1:
        score += 20

    return score


def find_best_strategic_move(bottles):
    """Choose the best move using heuristic scoring plus memory/reveal bonuses.

    Deadlock/loop control:
    - We score all valid moves.
    - We avoid moves whose simulated result repeats the current state (no-op).
    - We avoid moves whose simulated result is already in seen_state_signatures, unless no alternative exists.
    """
    candidates = []

    n = len(bottles)
    cur_sig = board_signature(state_board) if state_board is not None else None

    for i in range(n):
        if not bottles[i]:
            continue

        for j in range(n):
            if i == j:
                continue

            if not can_pour(bottles[i], bottles[j], i, j):
                continue

            base_score = score_strategic_move(bottles[i], bottles[j])
            score = base_score

            sim_sig = None
            if state_board is not None:
                tracked_color = bottle_contents.get(i + 1)
                sim = simulate_move_on_board(state_board, i + 1, j + 1, tracked_top_color=tracked_color)
                if sim is not None:
                    sim_sig = board_signature(sim)

                    # reveal bonus
                    before_unknowns = count_unknowns(state_board)
                    after_unknowns = count_unknowns(sim)
                    revealed = max(0, before_unknowns - after_unknowns)
                    score += revealed * 120  # stronger reveal preference

                    # no-op penalty
                    if cur_sig is not None and sim_sig == cur_sig:
                        score -= 2000

                    # loop penalty
                    if sim_sig in seen_state_signatures:
                        score -= 1200

            candidates.append((score, i + 1, j + 1, sim_sig))

    if not candidates:
        return None

    # Sort by score descending
    candidates.sort(key=lambda x: x[0], reverse=True)

    # Prefer a move that doesn't go to a seen signature
    for score, src, dst, sim_sig in candidates:
        if sim_sig is None:
            return {"from": src, "to": dst, "score": score}
        if sim_sig not in seen_state_signatures:
            return {"from": src, "to": dst, "score": score}

    # If all moves loop, return the best-scoring one anyway (deadlock handler will stop)
    score, src, dst, sim_sig = candidates[0]
    return {"from": src, "to": dst, "score": score}
def normal_solve(bottles):
    """Normal water sort solver without unknowns"""
    visited = set()
    solution = []
    
    def backtrack(board, moves_so_far, depth=0):
        if depth > MAX_DEPTH:
            return False
        
        board_key = serialize_board(board)
        if board_key in visited:
            return False
        visited.add(board_key)
        
        if is_solved(board):
            solution.extend(moves_so_far)
            return True
        
        for i in range(len(board)):
            if not board[i] or is_uniform_full(board[i]) or is_almost_full_same(board[i]):
                continue
            
            for j in range(len(board)):
                if i == j:
                    continue
                if can_pour(board[i], board[j], i, j):
                    board_copy = copy.deepcopy(board)
                    do_pour(board_copy, i, j)
                    moves_copy = moves_so_far + [{"from": i+1, "to": j+1}]
                    if backtrack(board_copy, moves_copy, depth+1):
                        return True
        
        return False
    
    if backtrack(bottles, []):
        return solution
    return None

def solve_puzzle(bottles):
    """Main solve: normal if no unknowns, strategic if unknowns present"""
    if has_unknowns(bottles):
        unknown_count = sum(b.count('?') for b in bottles)
        print(f"  Unknown segments: {unknown_count} - using strategic solving")
        move = find_best_strategic_move(bottles)
        if move:
            print(f"  Best move (score {move['score']}): Bottle {move['from']} -> {move['to']}")
            return [move]
        else:
            print("  No valid moves found")
            return None
    else:
        print("  No unknowns - using normal solver")
        return normal_solve(bottles)

# --------------------------
# EXECUTION
# --------------------------
def adb_tap(x, y):
    subprocess.run(
        ["adb", "shell", "input", "tap", str(x), str(y)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

def execute_moves(moves, bottle_coords):
    executed = []
    
    for i, move in enumerate(moves, start=1):
        src = move["from"]
        dst = move["to"]
        
        src_coord = bottle_coords.get(str(src))
        dst_coord = bottle_coords.get(str(dst))
        
        if not src_coord or not dst_coord:
            print("  ERROR: Missing coords")
            continue
        
        src_x, src_y = src_coord
        dst_x, dst_y = dst_coord
        
        print(f"  Executing: Bottle {src} -> {dst}")
        
        adb_tap(src_x, src_y)
        time.sleep(TAP_STEP_DELAY)
        adb_tap(dst_x, dst_y)
        apply_move_to_state(src, dst)
        # Remember resulting state to avoid loops
        if state_board is not None:
            seen_state_signatures.add(board_signature(state_board))
        time.sleep(MOVE_DELAY)
        
        tap_log.append({
            "move_index": len(all_moves_executed) + i,
            "from": src,
            "to": dst,
            "src_coord": [src_x, src_y],
            "dst_coord": [dst_x, dst_y]
        })
        executed.append(move)
    
    return executed

# --------------------------
# MAIN
# --------------------------
def main():
    global bottle_contents
    
    global _prev_unknowns, _noinfo_streak, _state_seen_counts
    print("="*60)
    print("ADAPTIVE WATER SORT SOLVER")
    print("="*60)
    print("Press Ctrl+C at any time to stop.")
    
    iteration = 0
    
    while iteration < MAX_ITERATIONS:
        iteration += 1
        print(f"\n{'='*60}")
        print(f"ITERATION {iteration}")
        print(f"{'='*60}")
        
        print("Detecting bottles...")
        bottle_data, bottle_coords = detect_and_analyze(force_last_two_empty=(iteration == 1))
        bottles = convert_to_solver_format(bottle_data)

        # Load saved learning on first iteration (helps hard-mode reveals)
        if "iteration" in locals() and iteration == 1 and state_board is not None:
            snap = load_learning_snapshot(len(state_board))
            if snap:
                apply_snapshot_to_state(snap)
                # Rebuild bottles after applying snapshot
                bottles = convert_to_solver_format(bottle_data)

        # Deadlock bookkeeping (loop / no-new-reveals detection)
        if state_board is not None:
            sig = board_signature(state_board)
            _state_seen_counts[sig] = _state_seen_counts.get(sig, 0) + 1

            unk = count_unknowns(state_board)
            if _prev_unknowns is None:
                _prev_unknowns = unk
            else:
                if unk >= _prev_unknowns:
                    _noinfo_streak += 1
                else:
                    _noinfo_streak = 0
                _prev_unknowns = unk

            if _state_seen_counts[sig] >= DEADLOCK_REPEAT_LIMIT or _noinfo_streak >= DEADLOCK_NOINFO_LIMIT:
                print("[DEADLOCK] Detected: repeating state / no new reveals.")
                save_learning_snapshot(reason="deadlock")
                sys.exit(0)


        
        print("\nCurrent state:")
        for idx, b in enumerate(bottles, 1):
            display = b if b else 'empty'
            tracked = f" [tracked: {bottle_contents[idx]}]" if idx in bottle_contents else ""
            print(f"  Bottle {idx}: {display}{tracked}")
        
        if is_solved(bottles):
            print("\n" + "="*60)
            print("PUZZLE SOLVED!")
            print("="*60)
            print(f"Total moves: {len(all_moves_executed)}")
            save_learning_snapshot(reason="solved")
            sys.exit(0)
        
        print("\nSolving...")
        moves = solve_puzzle(bottles)
        
        if not moves:
            print("\nNo solution found with current information.")
            save_learning_snapshot(reason="no_moves")
            sys.exit(0)
        
        print(f"\nExecuting {len(moves)} move(s)...")
        executed = execute_moves(moves, bottle_coords)
        all_moves_executed.extend(executed)
        
        # Update our tracking based on moves executed
        for move in executed:
            from_idx = move["from"] - 1
            to_idx = move["to"]
            
            # Get the color that was moved
            if from_idx < len(bottles) and bottles[from_idx]:
                color_moved = bottles[from_idx][0]
                
                # Track that this color is now in destination (only if destination was empty in our current detection)
                if to_idx not in bottle_contents:
                    # If we *think* destination is empty (no segments), then record the poured color.
                    # Otherwise, let detection drive tracking correction on next iteration.
                    if not bottles[to_idx - 1]:
                        bottle_contents[to_idx] = color_moved
                        print(f"  Tracked: Bottle {to_idx} now contains {color_moved}")
        
        # Wait for animation and re-scan
        print("\nWaiting for animation...")
        time.sleep(2.0)
    
    with open(TAP_LOG_PATH, "w") as f:
        json.dump(tap_log, f, indent=2)
    
    print(f"\nSession complete. Total moves: {len(all_moves_executed)}")

if __name__ == "__main__":
    main()
