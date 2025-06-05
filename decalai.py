# ── DecalAI.py ────────────────────────────────────────────────────────────────
import os
import sys
import json
import requests
import datetime
import socket

import numpy as np
import cv2
import fitz                     # PyMuPDF, for PDF → image rendering
import torch
import pandas as pd
import tkinter as tk
from tkinter import filedialog

# ── Segment‐Anything imports ─────────────────────────────────────────────────
# Install via `pip install segment-anything`
from segment_anything import sam_model_registry, SamPredictor

# ── IMPORT OUR HELPER FOR API‐KEY HANDLING ─────────────────────────────────────
from DecalAI_helper import get_valid_api_key


# ── CONFIGURATION ──────────────────────────────────────────────────────────────

# (1) API endpoint (we’ll fetch the key at runtime via helper)
API_ENDPOINT   = "https://hal4ecrr1tk.execute-api.us-east-1.amazonaws.com/prod/get_current_drawing"
API_KEY        = None    # ← will be populated by get_valid_api_key()

# (2) PDF rendering DPI (for “high‐resolution” BGR numpy array)
RENDER_DPI     = 300

# (3) SAM configuration
SAM_MODEL_TYPE = "vit_h"   # options: "vit_h", "vit_l", "vit_b" (match your checkpoint)
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"

# (4) Padding (as a fraction of bbox size)
CROP_PAD_PCT   = 0.05     # 5% padding on each side after we find the union bbox

# (5) SAM checkpoint location (will be overridden in main after folder selection)
SAM_CHECKPOINT = None     


# ── HELPER: Prompt for input sheet & output folder ─────────────────────────────
def ask_for_paths():
    """
    Opens two file dialogs:
      1) Excel/CSV sheet containing PART and TMS columns
      2) Output directory (where subfolders will be created)

    Returns:
      (sheet_path, base_output_dir)
    """
    root = tk.Tk()
    root.withdraw()

    # 1) Ask for the parts sheet
    sheet_path = filedialog.askopenfilename(
        title="Select parts sheet (Excel or CSV)",
        filetypes=[("Excel/CSV", "*.xlsx *.xls *.csv")]
    )
    if not sheet_path:
        print("No sheet selected; exiting.")
        sys.exit(0)

    # 2) Ask for the output folder
    out_root = filedialog.askdirectory(title="Select output folder")
    if not out_root:
        print("No output folder selected; exiting.")
        sys.exit(0)

    return sheet_path, out_root


# ── HELPER: Prepare subfolders under the chosen output directory ──────────────
def prepare_output_folders(base_output):
    """
    Given a user‐selected base_output directory, create (if needed):
      - base_output/temp_pdfs
      - base_output/outputs
      - base_output/sam_checkpoints

    Returns:
      (pdf_dir, output_dir, sam_ckpt_dir)
    """
    pdf_dir = os.path.join(base_output, "temp_pdfs")
    output_dir = os.path.join(base_output, "outputs")
    sam_ckpt_dir = os.path.join(base_output, "sam_checkpoints")

    for d in (pdf_dir, output_dir, sam_ckpt_dir):
        os.makedirs(d, exist_ok=True)

    return pdf_dir, output_dir, sam_ckpt_dir


# ── 1) Helper: Fetch PDF via signed‐URL API ─────────────────────────────────────
# ── DEBUG: Verify that the hostname actually resolves ─────────────────────────
    host = API_ENDPOINT.split("/")[2]  # e.g. "hal4ecrr1k.execute-api.us-east-1.amazonaws.com"
    try:
        addr = socket.getaddrinfo(host, 443)
        print(f"DEBUG: DNS lookup succeeded for {host} → {addr[0][4][0]}")
    except Exception as dns_err:
        print(f"DEBUG: DNS lookup failed for {host}: {dns_err}")
        return None
    # ──────────────────────────────────────────────────────────────────────────────

    # Now proceed with the POST, knowing the host is (or isn’t) resolving:
    try:
        response = requests.post(API_ENDPOINT, headers=headers, data=json.dumps(body), timeout=30)
        response.raise_for_status()
    except Exception as e:
        print(f"   · [ERROR] Failed to call API for '{part_number}': {e}")
        return None

    data = response.json()
    # IT might return either: {"url":"https://…"}  or  a bare URL string
    if isinstance(data, dict) and "url" in data:
        signed_url = data["url"]
    elif isinstance(data, str) and data.startswith("http"):
        signed_url = data
    else:
        print(f"   · [ERROR] Unexpected API response for '{part_number}': {data!r}")
        return None

    # (d) Download PDF from the signed URL
    try:
        dl = requests.get(signed_url, timeout=60)
        dl.raise_for_status()
    except Exception as e:
        print(f"   · [ERROR] Cannot download PDF for '{part_number}': {e}")
        return None

    # (e) Save the PDF locally
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{clean_part}_{ts}.pdf"
    local_path = os.path.join(pdf_dir, filename)
    with open(local_path, "wb") as f:
        f.write(dl.content)

    print(f"   · [API] Downloaded PDF for '{part_number}' → {local_path}")
    return local_path


# ── 2) Helper: Render first page of a PDF into a BGR image ───────────────────────
def render_pdf_page_to_bgr(pdf_path: str, dpi: int = RENDER_DPI) -> np.ndarray:
    """
    Uses PyMuPDF to open the first page of `pdf_path` at the specified DPI,
    then returns an H×W×3 BGR numpy array (suitable for cv2).
    """
    doc = fitz.open(pdf_path)
    page = doc.load_page(0)  # first page only
    zoom = dpi / 72  # MuPDF default is 72 DPI
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if pix.n == 4:
        # RGBA → BGR
        return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
    else:
        # RGB → BGR
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


# ── 3) Initialize the SAM model and predictor ───────────────────────────────────
def load_sam_model(sam_checkpoint: str, model_type: str = SAM_MODEL_TYPE, device: str = DEVICE):
    """
    Loads a pretrained Segment‐Anything (SAM) model from `sam_checkpoint`,
    returning a SamPredictor instance on the chosen device.
    """
    if not os.path.exists(sam_checkpoint):
        print(f"   · [ERROR] SAM checkpoint not found at: {sam_checkpoint}")
        sys.exit(1)

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device)
    predictor = SamPredictor(sam)
    print(f"   · Loaded SAM (type={model_type}) on {device}")
    return predictor


# ── 4) Helper: Get a single “largest object” mask & bounding box via SAM ────────
def segment_largest_object(sam_predictor: SamPredictor, image_bgr: np.ndarray) -> tuple:
    """
    Given a BGR image and a SamPredictor:
      1) convert to RGB for SAM
      2) predict a single mask (multimask_output=False)
      3) find the bounding box of the largest connected region in the mask
      4) return (x0, y0, x1, y1)

    Returns None if no mask was found.
    """
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    sam_predictor.set_image(image_rgb)

    masks, scores, logits = sam_predictor.predict(
        point_coords=None,
        point_labels=None,
        box=None,
        multimask_output=False
    )
    mask = masks[0]  # boolean array of shape (H, W)
    mask_uint8 = (mask.astype(np.uint8) * 255)

    # Find contours on this mask to get the bounding box of the largest area
    cnts, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None

    # Pick the contour with largest area
    c_max = max(cnts, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c_max)
    return (x, y, x + w, y + h)


# ── 5) Helper: Crop + pad a bounding box out of an image ───────────────────────
def crop_with_padding(image: np.ndarray, bbox: tuple, pad_pct: float = CROP_PAD_PCT) -> np.ndarray:
    """
    Given an image and bbox=(x0, y0, x1, y1),
    pad each side by `pad_pct * (min(width, height))`,
    then return the cropped sub-image.
    """
    img_h, img_w = image.shape[:2]
    x0, y0, x1, y1 = bbox
    rect_w = x1 - x0
    rect_h = y1 - y0
    pad = int(min(rect_w, rect_h) * pad_pct)

    x0p = max(x0 - pad, 0)
    y0p = max(y0 - pad, 0)
    x1p = min(x1 + pad, img_w)
    y1p = min(y1 + pad, img_h)
    return image[y0p:y1p, x0p:x1p]


# ── 6) Main processing for a single part number ────────────────────────────────
def process_part(part_number: str, tms: str, sam_predictor: SamPredictor,
                 pdf_dir: str, output_dir: str):
    """
    1) Fetch PDF via API  → local_pdf_path
    2) Render PDF → BGR image
    3) Run SAM to get a “largest object” bbox
    4) Crop + pad that bbox
    5) Save to output_dir as JPEG
    """
    print(f"[>] Processing part={part_number!r}, TMS={tms!r}")

    # 1) Fetch the PDF
    pdf_path = fetch_pdf_via_api(part_number, pdf_dir)
    if not pdf_path:
        print(f"   · [SKIP] No PDF for '{part_number}'")
        return

    # 2) Render first page of PDF to BGR image
    img_bgr = render_pdf_page_to_bgr(pdf_path, dpi=RENDER_DPI)

    # 3) Segment the largest object with SAM
    bbox = segment_largest_object(sam_predictor, img_bgr)
    if bbox is None:
        print(f"   · [WARN] SAM found no mask for '{part_number}'. Saving full‐page crop.")
        cropped = img_bgr.copy()
    else:
        # 4) Crop AND add padding
        cropped = crop_with_padding(img_bgr, bbox, pad_pct=CROP_PAD_PCT)
        print(f"   · [OK] Cropped bbox={bbox}, padded size={cropped.shape[1]}×{cropped.shape[0]}")

    # 5) Save the final crop
    clean_part = part_number.strip()
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_name = f"{tms}.{clean_part}.{timestamp}.jpg"
    out_path = os.path.join(output_dir, out_name)
    cv2.imwrite(out_path, cropped)
    print(f"   · [SAVED] → {out_path}\n")

    # Optionally remove the PDF to save space:
    try:
        os.remove(pdf_path)
    except Exception:
        pass


# ── 7) Entry Point: prompt for paths, load sheet, iterate over parts ──────────
def main():
    global API_KEY

    # (0) Obtain or validate the user’s API key via our helper
    try:
        API_KEY = get_valid_api_key()
    except Exception as err:
        print(f"Error: {err}")
        sys.exit(1)

    # (1) Ask user for input sheet & output folder
    sheet_path, base_output = ask_for_paths()

    # (2) Prepare subfolders under base_output
    pdf_dir, output_dir, sam_ckpt_dir = prepare_output_folders(base_output)

    # (3) Determine the SAM checkpoint location inside the repo's “models” folder
    global SAM_CHECKPOINT
    # ── locate the folder where DecalAI.py lives
    repo_root = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(repo_root, "models")
    # ── find any “.pth” in ./models
    ckpt_files = [f for f in os.listdir(models_dir) if f.lower().endswith(".pth")]
    if len(ckpt_files) == 0:
        print(f"   · [ERROR] No SAM checkpoint (.pth) found in {models_dir}")
        print("     Please download a SAM checkpoint and place it there (e.g. sam_vit_h_4b8939.pth).")
        sys.exit(1)
    elif len(ckpt_files) > 1:
        print(f"   · [WARNING] Multiple .pth files found in {models_dir}. Using the first one.")
    SAM_CHECKPOINT = os.path.join(models_dir, ckpt_files[0])
    
    # (4) Load the SAM predictor
    sam_predictor = load_sam_model(SAM_CHECKPOINT, model_type=SAM_MODEL_TYPE, device=DEVICE)

    # (5) Read the parts sheet (Excel or CSV)
    ext = os.path.splitext(sheet_path)[1].lower()
    if ext in (".xlsx", ".xls"):
        df = pd.read_excel(sheet_path, dtype=str)
    elif ext == ".csv":
        df = pd.read_csv(sheet_path, dtype=str)
    else:
        print(f"   · [ERROR] Unsupported file type: {ext}")
        sys.exit(1)

    df.columns = df.columns.str.upper()
    df.rename(columns={df.columns[0]: "PART", df.columns[1]: "TMS"}, inplace=True)
    df = df[df["PART"].notna()].copy()
    df["PART"] = df["PART"].str.strip()
    df["TMS"] = df["TMS"].astype(str).str.strip()

    # (6) Loop over each part
    for idx, row in df.iterrows():
        part_number = row["PART"]
        tms = row["TMS"]
        process_part(part_number, tms, sam_predictor, pdf_dir, output_dir)

    print("All done.")


if __name__ == "__main__":
    main()
