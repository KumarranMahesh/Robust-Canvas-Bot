"""
Linguistic Restoration Engine

Processes text-dominant images through a three-stage pipeline:
  1. OCR Detection & Recognition (EasyOCR)
  2. Spell Repair via statistical correction (pyspellchecker)
  3. Vector Typography Synthesis — generating robot-executable stroke paths

The vector synthesis module covers the full uppercase Latin alphabet (A-Z),
digits (0-9), and common punctuation, each defined as a set of skeletal
multi-point stroke paths suitable for robotic pen execution.
"""

import cv2
import numpy as np

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False

try:
    from spellchecker import SpellChecker
    SPELLCHECKER_AVAILABLE = True
except ImportError:
    SPELLCHECKER_AVAILABLE = False


# ─── Full Alphabet Vector Glyph Definitions ──────────────────────────────────
# Each glyph is defined as a list of stroke paths.
# Each stroke path is a list of (dx, dy) coordinates relative to the
# character's bounding box origin (top-left corner).
# Character cell: 20px wide, 30px tall. Inter-character spacing: 25px.

CHAR_WIDTH = 20
CHAR_HEIGHT = 30
CHAR_SPACING = 25

GLYPH_MAP = {
    'A': [
        [(0, 30), (10, 0), (20, 30)],      # two diagonal strokes
        [(5, 15), (15, 15)],                 # horizontal crossbar
    ],
    'B': [
        [(0, 0), (0, 30)],                  # vertical spine
        [(0, 0), (15, 0), (18, 5), (15, 15), (0, 15)],  # upper bump
        [(0, 15), (15, 15), (18, 20), (18, 25), (15, 30), (0, 30)],  # lower bump
    ],
    'C': [
        [(20, 5), (15, 0), (5, 0), (0, 5), (0, 25), (5, 30), (15, 30), (20, 25)],
    ],
    'D': [
        [(0, 0), (0, 30)],
        [(0, 0), (12, 0), (18, 5), (20, 15), (18, 25), (12, 30), (0, 30)],
    ],
    'E': [
        [(20, 0), (0, 0), (0, 30), (20, 30)],  # C-shape
        [(0, 15), (12, 15)],                      # middle bar
    ],
    'F': [
        [(0, 30), (0, 0), (20, 0)],   # vertical + top bar
        [(0, 15), (12, 15)],            # middle bar
    ],
    'G': [
        [(20, 5), (15, 0), (5, 0), (0, 5), (0, 25), (5, 30), (15, 30), (20, 25), (20, 15), (10, 15)],
    ],
    'H': [
        [(0, 0), (0, 30)],
        [(0, 15), (20, 15)],
        [(20, 0), (20, 30)],
    ],
    'I': [
        [(5, 0), (15, 0)],    # top serif
        [(10, 0), (10, 30)],   # vertical
        [(5, 30), (15, 30)],   # bottom serif
    ],
    'J': [
        [(5, 0), (20, 0)],
        [(15, 0), (15, 25), (10, 30), (5, 30), (0, 25)],
    ],
    'K': [
        [(0, 0), (0, 30)],
        [(20, 0), (0, 15)],
        [(0, 15), (20, 30)],
    ],
    'L': [
        [(0, 0), (0, 30), (20, 30)],
    ],
    'M': [
        [(0, 30), (0, 0), (10, 15), (20, 0), (20, 30)],
    ],
    'N': [
        [(0, 30), (0, 0), (20, 30), (20, 0)],
    ],
    'O': [
        [(5, 0), (15, 0), (20, 5), (20, 25), (15, 30), (5, 30), (0, 25), (0, 5), (5, 0)],
    ],
    'P': [
        [(0, 30), (0, 0), (15, 0), (20, 5), (20, 10), (15, 15), (0, 15)],
    ],
    'Q': [
        [(5, 0), (15, 0), (20, 5), (20, 25), (15, 30), (5, 30), (0, 25), (0, 5), (5, 0)],
        [(12, 22), (22, 32)],  # tail
    ],
    'R': [
        [(0, 30), (0, 0), (15, 0), (20, 5), (20, 10), (15, 15), (0, 15)],
        [(10, 15), (20, 30)],  # leg
    ],
    'S': [
        [(20, 5), (15, 0), (5, 0), (0, 5), (0, 10), (5, 15), (15, 15), (20, 20), (20, 25), (15, 30), (5, 30), (0, 25)],
    ],
    'T': [
        [(0, 0), (20, 0)],     # top bar
        [(10, 0), (10, 30)],   # vertical
    ],
    'U': [
        [(0, 0), (0, 25), (5, 30), (15, 30), (20, 25), (20, 0)],
    ],
    'V': [
        [(0, 0), (10, 30), (20, 0)],
    ],
    'W': [
        [(0, 0), (5, 30), (10, 15), (15, 30), (20, 0)],
    ],
    'X': [
        [(0, 0), (20, 30)],
        [(20, 0), (0, 30)],
    ],
    'Y': [
        [(0, 0), (10, 15)],
        [(20, 0), (10, 15)],
        [(10, 15), (10, 30)],
    ],
    'Z': [
        [(0, 0), (20, 0), (0, 30), (20, 30)],
    ],
    '0': [
        [(5, 0), (15, 0), (20, 5), (20, 25), (15, 30), (5, 30), (0, 25), (0, 5), (5, 0)],
        [(5, 5), (15, 25)],  # diagonal slash
    ],
    '1': [
        [(5, 5), (10, 0), (10, 30)],
        [(5, 30), (15, 30)],
    ],
    '2': [
        [(0, 5), (5, 0), (15, 0), (20, 5), (20, 10), (0, 30), (20, 30)],
    ],
    '3': [
        [(0, 5), (5, 0), (15, 0), (20, 5), (20, 10), (15, 15), (20, 20), (20, 25), (15, 30), (5, 30), (0, 25)],
    ],
    '4': [
        [(0, 0), (0, 15), (20, 15)],
        [(15, 0), (15, 30)],
    ],
    '5': [
        [(20, 0), (0, 0), (0, 15), (15, 15), (20, 20), (20, 25), (15, 30), (5, 30), (0, 25)],
    ],
    '6': [
        [(15, 0), (5, 0), (0, 5), (0, 25), (5, 30), (15, 30), (20, 25), (20, 20), (15, 15), (0, 15)],
    ],
    '7': [
        [(0, 0), (20, 0), (10, 30)],
    ],
    '8': [
        [(5, 0), (15, 0), (20, 5), (20, 10), (15, 15), (5, 15), (0, 10), (0, 5), (5, 0)],
        [(5, 15), (15, 15), (20, 20), (20, 25), (15, 30), (5, 30), (0, 25), (0, 20), (5, 15)],
    ],
    '9': [
        [(20, 15), (15, 15), (5, 15), (0, 10), (0, 5), (5, 0), (15, 0), (20, 5), (20, 25), (15, 30), (5, 30)],
    ],
    ' ': [],  # space — no strokes, just advance
    '.': [
        [(8, 28), (12, 28), (12, 30), (8, 30), (8, 28)],
    ],
    ',': [
        [(10, 26), (10, 32)],
    ],
    '-': [
        [(3, 15), (17, 15)],
    ],
    '!': [
        [(10, 0), (10, 22)],
        [(10, 27), (10, 30)],
    ],
    '?': [
        [(0, 5), (5, 0), (15, 0), (20, 5), (20, 10), (10, 15), (10, 22)],
        [(10, 27), (10, 30)],
    ],
}


class LinguisticRestoration:
    def __init__(self):
        print("Loading Linguistic Restoration Engine...")

        if EASYOCR_AVAILABLE:
            self.reader = easyocr.Reader(['en'], gpu=True)
        else:
            print("  Warning: EasyOCR not available. OCR detection disabled.")
            self.reader = None

        if SPELLCHECKER_AVAILABLE:
            self.spell = SpellChecker()
        else:
            print("  Warning: pyspellchecker not available. Spell correction disabled.")
            self.spell = None

    def llm_spell_check_proxy(self, text):
        """
        Lightweight spell correction proxy using pyspellchecker.

        Corrects misspellings by finding the most probable candidate
        within edit distance 2, based on word frequency statistics.
        """
        if self.spell is None:
            return text

        words = text.split()
        corrected = []
        for w in words:
            w_lower = w.lower()
            if w_lower in self.spell.unknown([w_lower]):
                cand = self.spell.correction(w_lower)
                corrected.append(cand if cand is not None else w)
            else:
                corrected.append(w)
        return " ".join(corrected)

    @staticmethod
    def generate_vector_synthesis(text, bbox):
        """
        Convert a corrected text string into robot-executable vector stroke paths
        using an Affine transformation mapped to the original OCR bounding box.

        Each character is mapped to a set of skeletal line segments defined in
        GLYPH_MAP. Transformations adapt to exactly match position, scale, and tilt.

        Args:
            text: The corrected text string to synthesize.
            bbox: The bounding box from EasyOCR (list of 4 points: [tl, tr, br, bl]).

        Returns:
            List of stroke paths, where each path is a list of (x, y) tuples.
        """
        paths = []
        if len(text) == 0:
            return paths

        # Define bounds of the standard text line
        std_width = max(1, len(text) * CHAR_SPACING - CHAR_SPACING + CHAR_WIDTH)
        std_height = CHAR_HEIGHT

        # Source coordinates [Top-Left, Top-Right, Bottom-Left]
        pts1 = np.float32([[0, 0], [std_width, 0], [0, std_height]])
        # Target bounding box coordinates
        pts2 = np.float32([bbox[0], bbox[1], bbox[3]])

        # Affine mapping matrix 
        M = cv2.getAffineTransform(pts1, pts2)

        curr_x = 0
        for char in text.upper():
            glyph = GLYPH_MAP.get(char, None)

            if glyph is None:
                # Fallback: draw a small box for unknown characters
                stroke = [
                    (2, 2),
                    (CHAR_WIDTH - 2, 2),
                    (CHAR_WIDTH - 2, CHAR_HEIGHT - 2),
                    (2, CHAR_HEIGHT - 2),
                    (2, 2)
                ]
                mapped_stroke = []
                for dx, dy in stroke:
                    pt = np.array([curr_x + dx, dy, 1.0])
                    mapped_pt = M.dot(pt)
                    mapped_stroke.append((mapped_pt[0], mapped_pt[1]))
                paths.append(mapped_stroke)
            else:
                for stroke in glyph:
                    mapped_stroke = []
                    for dx, dy in stroke:
                        pt = np.array([curr_x + dx, dy, 1.0])
                        mapped_pt = M.dot(pt)
                        mapped_stroke.append((mapped_pt[0], mapped_pt[1]))
                    paths.append(mapped_stroke)

            curr_x += CHAR_SPACING

        return paths

    def process_image(self, img_bgr):
        """
        Execute the full text pipeline: Detect -> Recognise & Repair -> Vector Synthesis.

        Returns:
            List of stroke paths for robotic execution.
        """
        if self.reader is None:
            print("  OCR not available — returning empty paths.")
            return []

        results = self.reader.readtext(img_bgr)
        all_paths = []

        for (bbox, text, prob) in results:
            if prob < 0.1:
                continue

            corrected_text = self.llm_spell_check_proxy(text)
            print(f"  Detected: '{text}' -> Repaired: '{corrected_text}' (conf: {prob:.2f})")

            paths = self.generate_vector_synthesis(corrected_text, bbox)
            all_paths.extend(paths)

        return all_paths


if __name__ == "__main__":
    # Test vector synthesis
    test_bbox = [[10, 10], [210, 10], [210, 40], [10, 40]]
    paths = LinguisticRestoration.generate_vector_synthesis("HELLO WORLD", test_bbox)
    print(f"Generated {len(paths)} stroke paths for 'HELLO WORLD'")
    for i, p in enumerate(paths[:5]):
        print(f"  Stroke {i}: {p}")
