#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
automatic_mask.py
Single-file Tkinter app for mask creation + CSV processing.

Per-worm outputs kept:
  - worm_k_x, worm_k_y
  - worm_k_nose_x, worm_k_nose_y (if present in input)
  - worm_k_centroid_on_food (per-frame centroid vs. mask)
  - worm_k_food_encounter   (per-frame NOSE vs. mask)

• Nose x/y are optional in the CSV and ARE saved in the output if present.
• Auto-saves food boundary coordinates (JSON) alongside CSV when saving results.
• Auto-loads a SAM checkpoint from ./segmentation next to this script (if available).
• Works without SAM installed (you can load a mask PNG and process CSV).
"""

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import cv2
import numpy as np
from PIL import Image, ImageTk
import os
from pathlib import Path
import pandas as pd
import re
import threading
import queue

# Optional: Segment Anything (SAM). App runs fine if not installed.
try:
    from segment_anything import sam_model_registry, SamPredictor
    import torch
    SAM_AVAILABLE = True
except Exception:
    SAM_AVAILABLE = False
    sam_model_registry = {}
    SamPredictor = None
    torch = None
    print("SAM not installed. You can still load a mask PNG and process CSV.")

class AutoMaskNoseApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Automatic Mask + Nose Encounter")
        self.root.geometry("1400x900")

        # Image / mask state
        self.image = None
        self.original_image = None
        self.display_image = None
        self.current_mask = None  # binary (0/1) or bool
        self.sam_predictor = None

        # Click annotations for SAM
        self.positive_points = []
        self.negative_points = []
        self.current_mode = "positive"

        # CSV state
        self.csv_data = None
        self.csv_file_path = None
        self.processed_data = None
        self.worm_ids = []
        self.worm_ids_nose = []
        self.has_nose_columns = False

        # UI / display
        self.canvas_size = (800, 600)
        self.scale_factor = 1.0

        # Progress updates
        self.progress_queue = queue.Queue()

        # UI
        self._build_ui()

        # Auto-load a SAM checkpoint from ./segmentation if present (non-blocking)
        self._auto_load_sam()

    # ---------- UI ----------
    def _build_ui(self):
        main = ttk.Frame(self.root, padding=10)
        main.grid(row=0, column=0, sticky="nsew")
        self.root.rowconfigure(0, weight=1)
        self.root.columnconfigure(0, weight=1)
        main.columnconfigure(0, weight=1)
        main.columnconfigure(1, weight=1)
        main.rowconfigure(1, weight=1)

        controls = ttk.LabelFrame(main, text="Controls", padding=10)
        controls.grid(row=0, column=0, columnspan=2, sticky="ew")

        # SAM, image, mask
        ttk.Label(controls, text="Segmentation / Mask:").grid(row=0, column=0, sticky="w")
        ttk.Button(controls, text="Load SAM Model", command=self._load_sam_threaded,
                   state=("normal" if not SAM_AVAILABLE else "normal")).grid(row=0, column=1, padx=5)
        ttk.Button(controls, text="Load Image", command=self._load_image_threaded).grid(row=0, column=2, padx=5)
        ttk.Button(controls, text="Clear Points", command=self._clear_points).grid(row=0, column=3, padx=5)
        ttk.Button(controls, text="Save Mask", command=self._save_mask).grid(row=0, column=4, padx=5)
        ttk.Button(controls, text="Load Mask (PNG)", command=self._load_mask_png).grid(row=0, column=5, padx=5)
        ttk.Button(controls, text="View Boundary JSON", command=self._load_and_view_boundary).grid(row=0, column=6, padx=5)

        # Mode
        self.mode_var = tk.StringVar(value="positive")
        ttk.Label(controls, text="Click Mode:").grid(row=1, column=0, sticky="w", pady=(8,0))
        ttk.Radiobutton(controls, text="Add", value="positive", variable=self.mode_var,
                        command=self._set_mode).grid(row=1, column=1, sticky="w", pady=(8,0))
        ttk.Radiobutton(controls, text="Remove", value="negative", variable=self.mode_var,
                        command=self._set_mode).grid(row=1, column=2, sticky="w", pady=(8,0))

        # CSV
        ttk.Label(controls, text="CSV:").grid(row=2, column=0, sticky="w", pady=(8,0))
        ttk.Button(controls, text="Load CSV", command=self._load_csv).grid(row=2, column=1, padx=5, pady=(8,0))
        ttk.Button(controls, text="Process CSV", command=self._process_csv).grid(row=2, column=2, padx=5, pady=(8,0))
        ttk.Button(controls, text="Save Results", command=self._save_results).grid(row=2, column=3, padx=5, pady=(8,0))

        # Mask interpretation: which color is food?
        self.white_is_food = tk.BooleanVar(value=False)
        ttk.Checkbutton(controls, text="White area = ON food", variable=self.white_is_food).grid(row=2, column=4, padx=10, pady=(8,0))

        # Progress
        self.progress_var = tk.StringVar(value="Ready")
        prog = ttk.Frame(controls)
        prog.grid(row=3, column=0, columnspan=6, sticky="ew", pady=(10,0))
        prog.columnconfigure(0, weight=1)
        self.progress_bar = ttk.Progressbar(prog, mode="indeterminate")
        self.progress_bar.grid(row=0, column=0, sticky="ew", padx=(0,10))
        ttk.Label(prog, textvariable=self.progress_var).grid(row=0, column=1, sticky="e")

        # Content layout
        content = ttk.Frame(main)
        content.grid(row=1, column=0, columnspan=2, sticky="nsew", pady=(10,0))
        content.columnconfigure(0, weight=2)
        content.columnconfigure(1, weight=1)
        content.rowconfigure(0, weight=1)

        # Canvas
        canvas_frame = ttk.LabelFrame(content, text="Image Canvas", padding=5)
        canvas_frame.grid(row=0, column=0, sticky="nsew", padx=(0,5))
        self.canvas = tk.Canvas(canvas_frame, bg="white", width=self.canvas_size[0], height=self.canvas_size[1])
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Button-1>", self._on_canvas_click)

        # Info panel
        info = ttk.LabelFrame(content, text="Information", padding=10)
        info.grid(row=0, column=1, sticky="nsew")
        info.columnconfigure(0, weight=1)
        info.rowconfigure(1, weight=1)
        info.rowconfigure(3, weight=1)

        instructions = (
            "Workflow:\n"
            " 1) Load image\n"
            " 2) (Optional) Load SAM or load an existing mask (PNG)\n"
            " 3) Click to add/remove to/from mask (if SAM loaded)\n"
            " 4) Load CSV\n"
            " 5) Process → Save (auto-saves boundary JSON)\n\n"
            "CSV columns expected:\n"
            " • frame\n"
            " • worm_k_x, worm_k_y\n"
            " • optional: worm_k_nose_x, worm_k_nose_y\n\n"
            "Boundary: ~3px buffer around food edge\n"
        )
        ttk.Label(info, text=instructions, justify="left", wraplength=280).grid(row=0, column=0, sticky="nw")

        ttk.Label(info, text="CSV Info:", font=('TkDefaultFont', 9, 'bold')).grid(row=1, column=0, sticky="w", pady=(10,5))
        self.csv_info = tk.Text(info, wrap="word", height=12, width=35)
        self.csv_info_scroll = ttk.Scrollbar(info, orient="vertical", command=self.csv_info.yview)
        self.csv_info.configure(yscrollcommand=self.csv_info_scroll.set)
        self.csv_info.grid(row=2, column=0, sticky="nsew")
        self.csv_info_scroll.grid(row=2, column=1, sticky="ns")

        ttk.Label(info, text="Results:", font=('TkDefaultFont', 9, 'bold')).grid(row=3, column=0, sticky="w", pady=(10,5))
        self.results_info = tk.Text(info, wrap="word", height=12, width=35)
        self.results_scroll = ttk.Scrollbar(info, orient="vertical", command=self.results_info.yview)
        self.results_info.configure(yscrollcommand=self.results_scroll.set)
        self.results_info.grid(row=4, column=0, sticky="nsew")
        self.results_scroll.grid(row=4, column=1, sticky="ns")

        # Start progress pump
        self._check_progress_queue()

    # ---------- Progress pump ----------
    def _check_progress_queue(self):
        try:
            while True:
                msg = self.progress_queue.get_nowait()
                if msg == "PROGRESS_START":
                    self.progress_bar.start()
                elif msg == "PROGRESS_STOP":
                    self.progress_bar.stop()
                else:
                    self.progress_var.set(str(msg))
        except queue.Empty:
            pass
        self.root.after(100, self._check_progress_queue)

    # ---------- SAM ----------
    def _auto_load_sam(self):
        """Auto-load a SAM checkpoint from ./segmentation next to this script, if present."""
        if not SAM_AVAILABLE:
            self.progress_var.set("SAM not installed (optional).")
            return
        try:
            script_dir = Path(__file__).resolve().parent
            seg_dir = script_dir / "segmentation"
            # Prefer the smaller/faster B model if multiple are present
            preferred = [
                "sam_vit_b_01ec64.pth",
                "sam_vit_l_0b3195.pth",
                "sam_vit_h_4b8939.pth",
            ]
            checkpoint = None
            for fname in preferred:
                p = seg_dir / fname
                if p.exists():
                    checkpoint = str(p)
                    break
            # If not found in preferred list, fall back to any .pth in the folder
            if checkpoint is None and seg_dir.exists():
                pths = sorted(seg_dir.glob("*.pth"))
                if pths:
                    checkpoint = str(pths[0])

            if not checkpoint:
                self.progress_var.set("No SAM checkpoint found in ./segmentation (optional).")
                return

            def load_model():
                try:
                    self.progress_queue.put("PROGRESS_START")
                    self.progress_queue.put(f"Auto-loading SAM from {Path(checkpoint).name}...")
                    if "vit_h" in checkpoint:
                        model_type = "vit_h"
                    elif "vit_l" in checkpoint:
                        model_type = "vit_l"
                    else:
                        model_type = "vit_b"
                    # Initialize model WITHOUT passing checkpoint so we control torch.load defaults
                    model = sam_model_registry[model_type]()
                    # Robust load across torch versions (2.6+ default changed to weights_only=True)
                    try:
                        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
                    except TypeError:
                        # Older torch versions don't accept weights_only kwarg
                        state = torch.load(checkpoint, map_location="cpu")
                    # Unwrap common wrappers
                    if isinstance(state, dict) and "state_dict" in state:
                        state = state["state_dict"]
                    missing, unexpected = model.load_state_dict(state, strict=False)
                    if missing or unexpected:
                        self.progress_queue.put(f"SAM state_dict: missing={len(missing)} unexpected={len(unexpected)}")
                    device = "cuda" if (torch and torch.cuda.is_available()) else "cpu"
                    self.progress_queue.put(f"Moving SAM to {device.upper()}...")
                    model.to(device=device)
                    self.sam_predictor = SamPredictor(model)
                    self.progress_queue.put(f"SAM loaded: {model_type} ({device.upper()})")
                except Exception as e:
                    self.progress_queue.put(f"SAM auto-load failed: {e}")
                finally:
                    self.progress_queue.put("PROGRESS_STOP")

            threading.Thread(target=load_model, daemon=True).start()
        except Exception as e:
            self.progress_var.set(f"SAM auto-load error: {e}")

    def _load_sam_threaded(self):
        if not SAM_AVAILABLE:
            messagebox.showerror("Error", "SAM not installed:\n  pip install git+https://github.com/facebookresearch/segment-anything.git")
            return
        def load_model():
            self.progress_queue.put("PROGRESS_START")
            path = filedialog.askopenfilename(title="Select SAM checkpoint (*.pth)",
                                              initialdir=str((Path(__file__).resolve().parent / "segmentation").resolve()),
                                              filetypes=[("PyTorch checkpoint", "*.pth"), ("All files", "*.*")])
            if not path:
                self.progress_queue.put("PROGRESS_STOP")
                return
            try:
                if "vit_h" in path:
                    model_type = "vit_h"
                elif "vit_l" in path:
                    model_type = "vit_l"
                else:
                    model_type = "vit_b"
                model = sam_model_registry[model_type]()
                try:
                    state = torch.load(path, map_location="cpu", weights_only=False)
                except TypeError:
                    state = torch.load(path, map_location="cpu")
                if isinstance(state, dict) and "state_dict" in state:
                    state = state["state_dict"]
                missing, unexpected = model.load_state_dict(state, strict=False)
                if missing or unexpected:
                    self.progress_queue.put(f"SAM state_dict: missing={len(missing)} unexpected={len(unexpected)}")
                device = "cuda" if (torch and torch.cuda.is_available()) else "cpu"
                model.to(device=device)
                self.sam_predictor = SamPredictor(model)
                self.progress_queue.put(f"SAM loaded: {model_type} ({device.upper()})")
            except Exception as e:
                self.progress_queue.put(f"Failed to load SAM: {e}")
                messagebox.showerror("Error", f"Failed to load SAM:\n{e}")
            finally:
                self.progress_queue.put("PROGRESS_STOP")
        threading.Thread(target=load_model, daemon=True).start()

    # ---------- Image & Mask ----------
    def _load_image_threaded(self):
        def load_image():
            self.progress_queue.put("PROGRESS_START")
            path = filedialog.askopenfilename(title="Select image",
                                              filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"), ("All files", "*.*")])
            if not path:
                self.progress_queue.put("PROGRESS_STOP")
                return
            try:
                img_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
                if img_bgr is None:
                    raise RuntimeError("Failed to load image")
                img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                self.original_image = img.copy()
                # Downscale for display if huge
                h, w = img.shape[:2]
                max_dim = 1280
                if max(h, w) > max_dim:
                    if h >= w:
                        new_h = max_dim
                        new_w = int(w * (max_dim / h))
                    else:
                        new_w = max_dim
                        new_h = int(h * (max_dim / w))
                    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
                self.image = img
                if self.sam_predictor is not None:
                    self.sam_predictor.set_image(self.image)
                self._clear_points()
                self.root.after(0, self._display_image_on_canvas)
                self.progress_queue.put(f"Image loaded: {Path(path).name} ({self.image.shape[1]}x{self.image.shape[0]})")
            except Exception as e:
                self.progress_queue.put(f"Load image failed: {e}")
                messagebox.showerror("Error", f"Failed to load image:\n{e}")
            finally:
                self.progress_queue.put("PROGRESS_STOP")
        threading.Thread(target=load_image, daemon=True).start()

    def _display_image_on_canvas(self):
        if self.image is None:
            return
        h, w = self.image.shape[:2]
        scale_w = self.canvas_size[0] / w
        scale_h = self.canvas_size[1] / h
        self.scale_factor = min(scale_w, scale_h, 1.0)
        disp_w = int(w * self.scale_factor)
        disp_h = int(h * self.scale_factor)
        disp = cv2.resize(self.image, (disp_w, disp_h))

        # Overlay mask
        if self.current_mask is not None:
            mask_resized = cv2.resize(self.current_mask.astype(np.uint8), (disp_w, disp_h), interpolation=cv2.INTER_NEAREST)
            overlay = np.zeros_like(disp)
            if self.white_is_food.get():
                overlay[mask_resized > 0] = [0, 255, 0]
            else:
                overlay[mask_resized > 0] = [255, 0, 0]
            disp = cv2.addWeighted(disp, 0.7, overlay, 0.3, 0)

        # Points
        for x, y in self.positive_points:
            cx, cy = int(x * self.scale_factor), int(y * self.scale_factor)
            cv2.circle(disp, (cx, cy), 5, (0, 255, 0), -1)
        for x, y in self.negative_points:
            cx, cy = int(x * self.scale_factor), int(y * self.scale_factor)
            cv2.circle(disp, (cx, cy), 5, (255, 0, 0), -1)

        pil = Image.fromarray(disp)
        self.display_image = ImageTk.PhotoImage(pil)
        self.canvas.delete("all")
        cx = self.canvas_size[0] // 2
        cy = self.canvas_size[1] // 2
        self.canvas.create_image(cx, cy, image=self.display_image)

    def _on_canvas_click(self, event):
        if self.image is None:
            messagebox.showwarning("Warning", "Load an image first.")
            return
        if self.sam_predictor is None:
            messagebox.showwarning("Warning", "Load a SAM model or load a pre-made mask PNG.")
            return
        cx = self.canvas_size[0] // 2
        cy = self.canvas_size[1] // 2
        h, w = self.image.shape[:2]
        disp_w = int(w * self.scale_factor)
        disp_h = int(h * self.scale_factor)
        off_x = event.x - cx
        off_y = event.y - cy
        img_x = (disp_w // 2 + off_x) / self.scale_factor
        img_y = (disp_h // 2 + off_y) / self.scale_factor
        if 0 <= img_x < w and 0 <= img_y < h:
            if self.current_mode == "positive":
                self.positive_points.append((img_x, img_y))
            else:
                self.negative_points.append((img_x, img_y))
            self._generate_mask()

    def _generate_mask(self):
        if not SAM_AVAILABLE or self.sam_predictor is None:
            messagebox.showwarning("Warning", "SAM not available.")
            return
        if not self.positive_points and not self.negative_points:
            self.current_mask = None
            self._display_image_on_canvas()
            return
        try:
            pts = []
            labels = []
            for p in self.positive_points:
                pts.append(p); labels.append(1)
            for p in self.negative_points:
                pts.append(p); labels.append(0)
            pts = np.array(pts); labels = np.array(labels)
            masks, scores, logits = self.sam_predictor.predict(
                point_coords=pts,
                point_labels=labels,
                multimask_output=False
            )
            self.current_mask = masks[0]
            self._display_image_on_canvas()
        except Exception as e:
            messagebox.showerror("Error", f"Mask generation failed:\n{e}")

    def _clear_points(self):
        self.positive_points.clear()
        self.negative_points.clear()
        self.current_mask = None
        if self.image is not None:
            self._display_image_on_canvas()
        self.progress_var.set("Cleared points.")

    def _save_mask(self):
        if self.current_mask is None:
            messagebox.showwarning("Warning", "No mask to save.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".png",
                                            filetypes=[("PNG", "*.png"), ("All files", "*.*")])
        if not path:
            return
        try:
            if self.original_image is not None:
                oh, ow = self.original_image.shape[:2]
                mask = cv2.resize(self.current_mask.astype(np.uint8), (ow, oh), interpolation=cv2.INTER_NEAREST)
            else:
                mask = self.current_mask.astype(np.uint8)
            cv2.imwrite(path, (mask * 255).astype(np.uint8))
            self.progress_var.set(f"Mask saved: {Path(path).name}")
        except Exception as e:
            messagebox.showerror("Error", f"Saving mask failed:\n{e}")

    def _load_mask_png(self):
        path = filedialog.askopenfilename(title="Load Mask PNG",
                                          filetypes=[("PNG", "*.png"), ("All files", "*.*")])
        if not path:
            return
        try:
            mask_img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if mask_img is None:
                raise RuntimeError("Failed to load mask image")
            _, mask_bin = cv2.threshold(mask_img, 127, 1, cv2.THRESH_BINARY)
            self.current_mask = mask_bin.astype(np.uint8)
            self.progress_var.set(f"Loaded mask: {Path(path).name}")
            if self.image is not None:
                self._display_image_on_canvas()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load mask:\n{e}")

    def _load_and_view_boundary(self):
        """Load boundary JSON and display it in a new window."""
        path = filedialog.askopenfilename(title="Load Boundary JSON",
                                          filetypes=[("JSON", "*.json"), ("All files", "*.*")])
        if not path:
            return
        try:
            import json
            with open(path, 'r') as f:
                data = json.load(f)
            
            # Extract boundary coordinates
            boundary_coords = data['boundary_region']['coordinates']
            contour_coords = data['food_contour']['coordinates']
            dims = data['mask_dimensions']
            metrics = data['food_metrics']
            
            if len(boundary_coords) == 0 and len(contour_coords) == 0:
                messagebox.showwarning("Warning", "No boundary data in JSON")
                return
            
            # Create new window
            boundary_window = tk.Toplevel(self.root)
            boundary_window.title(f"Boundary Visualization - {Path(path).name}")
            boundary_window.geometry("1000x800")
            
            # Info panel
            info_frame = ttk.Frame(boundary_window, padding=10)
            info_frame.pack(side="top", fill="x")
            
            info_text = (
                f"Dimensions: {dims['width']} x {dims['height']} | "
                f"Food Area: {metrics['area_percentage']:.1f}% | "
                f"Boundary Pixels: {len(boundary_coords):,} | "
                f"Contour Points: {len(contour_coords):,}"
            )
            ttk.Label(info_frame, text=info_text).pack()
            
            # Canvas for visualization
            canvas = tk.Canvas(boundary_window, bg="white", width=900, height=700)
            canvas.pack(side="top", fill="both", expand=True, padx=10, pady=10)
            
            # Calculate scaling to fit canvas
            canvas_w = 900
            canvas_h = 700
            scale_x = canvas_w / dims['width']
            scale_y = canvas_h / dims['height']
            scale = min(scale_x, scale_y, 1.0)  # Don't upscale
            
            offset_x = (canvas_w - dims['width'] * scale) / 2
            offset_y = (canvas_h - dims['height'] * scale) / 2
            
            # Draw boundary pixels as red dots
            if len(boundary_coords) > 0:
                for x, y in boundary_coords[::10]:  # Subsample for performance
                    sx = offset_x + x * scale
                    sy = offset_y + y * scale
                    canvas.create_oval(sx-1, sy-1, sx+1, sy+1, fill="red", outline="red")
            
            # Draw contour as green line
            if len(contour_coords) > 1:
                points = []
                for x, y in contour_coords:
                    sx = offset_x + x * scale
                    sy = offset_y + y * scale
                    points.extend([sx, sy])
                # Close the contour
                points.extend([offset_x + contour_coords[0][0] * scale, 
                              offset_y + contour_coords[0][1] * scale])
                canvas.create_line(points, fill="green", width=2)
            
            # Draw centroid as blue X
            cx = offset_x + metrics['centroid']['x'] * scale
            cy = offset_y + metrics['centroid']['y'] * scale
            size = 10
            canvas.create_line(cx-size, cy-size, cx+size, cy+size, fill="blue", width=3)
            canvas.create_line(cx-size, cy+size, cx+size, cy-size, fill="blue", width=3)
            
            # Legend
            legend_y = 20
            canvas.create_oval(20, legend_y, 24, legend_y+4, fill="red", outline="red")
            canvas.create_text(30, legend_y+2, text="Boundary pixels", anchor="w", fill="red")
            
            canvas.create_line(20, legend_y+20, 30, legend_y+20, fill="green", width=2)
            canvas.create_text(35, legend_y+20, text="Food contour", anchor="w", fill="green")
            
            canvas.create_line(15, legend_y+40, 25, legend_y+50, fill="blue", width=2)
            canvas.create_line(15, legend_y+50, 25, legend_y+40, fill="blue", width=2)
            canvas.create_text(30, legend_y+45, text="Centroid", anchor="w", fill="blue")
            
            self.progress_var.set(f"Boundary loaded: {Path(path).name}")
            
        except Exception as e:
            import traceback
            messagebox.showerror("Error", f"Failed to load boundary:\n{e}\n\n{traceback.format_exc()}")

    def _set_mode(self):
        self.current_mode = self.mode_var.get()
        self.progress_var.set("Mode: Add" if self.current_mode == "positive" else "Mode: Remove")

    # ---------- CSV ----------
    def _load_csv(self):
        path = filedialog.askopenfilename(title="Select CSV", filetypes=[("CSV", "*.csv"), ("All files", "*.*")])
        if not path:
            return
        try:
            self.csv_data = pd.read_csv(path)
            self.csv_file_path = path
            self._analyze_csv()
            self.progress_var.set(f"CSV loaded: {os.path.basename(path)} ({len(self.csv_data)} rows)")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load CSV:\n{e}")

    def _analyze_csv(self):
        if self.csv_data is None:
            return
        self.csv_info.config(state="normal")
        self.csv_info.delete("1.0", "end")
        info = []
        info.append(f"Rows: {len(self.csv_data)}\n")
        info.append(f"Columns: {len(self.csv_data.columns)}\n\n")

        worm_ids = set()
        nose_ids = set()
        coord_cols = []
        for col in self.csv_data.columns:
            if col.lower() != "frame" and ('_x' in col or '_y' in col):
                coord_cols.append(col)
                base = '_'.join(col.split('_')[:-1])
                if base.endswith('nose'):
                    nose_ids.add(base[:-5])  # remove '_nose'
                else:
                    worm_ids.add(base)

        def nkey(s):
            return [int(t) if t.isdigit() else t for t in re.split('([0-9]+)', s)]

        self.worm_ids = sorted(worm_ids, key=nkey)
        self.worm_ids_nose = sorted([w for w in self.worm_ids if w in nose_ids], key=nkey)
        self.has_nose_columns = len(self.worm_ids_nose) > 0

        info.append(f"Tracked objects: {len(self.worm_ids)}\n")
        if self.has_nose_columns:
            info.append(f"Nose coords for {len(self.worm_ids_nose)} objects.\n")
        else:
            info.append("No nose coord columns detected.\n")

        info.append("\nPreview (first 3 rows):\n")
        for i in range(min(3, len(self.csv_data))):
            row = self.csv_data.iloc[i]
            fields = []
            for c in coord_cols[:8]:
                v = row[c]
                if pd.notna(v):
                    try:
                        fields.append(f"{c}={float(v):.1f}")
                    except Exception:
                        fields.append(f"{c}={v}")
            info.append("  " + ", ".join(fields) + ("\n" if i < 2 else ""))

        self.csv_info.insert("end", "".join(info))
        self.csv_info.config(state="disabled")

    # Vectorized per-series mask test (1=on food, 0=off, NaN for missing)
    def _series_on_food(self, xs, ys, mask_array):
        arr_x = xs.to_numpy(dtype='float64', copy=False)
        arr_y = ys.to_numpy(dtype='float64', copy=False)

        valid = ~(pd.isna(arr_x) | pd.isna(arr_y))
        xi = np.rint(arr_x[valid]).astype(np.int64, copy=False)
        yi = np.rint(arr_y[valid]).astype(np.int64, copy=False)

        h, w = mask_array.shape[:2]
        inb = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)

        out = np.full(arr_x.shape, np.nan, dtype=float)
        tmp = np.zeros(np.count_nonzero(valid), dtype=float)  # default off-food for valid-but-OOB

        if np.any(inb):
            mv = mask_array[yi[inb], xi[inb]]
            if self.white_is_food.get():
                vals = (mv > 0).astype(float)   # white==food
            else:
                vals = (mv == 0).astype(float)  # black==food
            tmp[inb] = vals

        out[valid] = tmp
        return out

    def _process_csv(self):
        if self.csv_data is None:
            messagebox.showwarning("Warning", "Load a CSV first.")
            return
        if self.current_mask is None:
            messagebox.showwarning("Warning", "Create or load a mask first.")
            return
        try:
            self.results_info.config(state="normal")
            self.results_info.delete("1.0", "end")

            # Use mask at original image size if available
            if self.original_image is not None:
                oh, ow = self.original_image.shape[:2]
                mask_array = cv2.resize(self.current_mask.astype(np.uint8), (ow, oh),
                                        interpolation=cv2.INTER_NEAREST)
            else:
                mask_array = self.current_mask.astype(np.uint8)

            df = self.csv_data.copy()
            lines = []

            for wid in self.worm_ids:
                x_col = f"{wid}_x"
                y_col = f"{wid}_y"
                nose_x_col = f"{wid}_nose_x"
                nose_y_col = f"{wid}_nose_y"

                cent_col = f"{wid}_centroid_on_food"   # kept
                enc_col  = f"{wid}_food_encounter"     # kept (per-frame NOSE)

                # Centroid per-frame on food
                if x_col in df.columns and y_col in df.columns:
                    df[cent_col] = self._series_on_food(df[x_col], df[y_col], mask_array)
                else:
                    df[cent_col] = np.nan

                # Nose per-frame encounter on food (not sticky). If no nose columns, NaN.
                have_nose = (nose_x_col in df.columns and nose_y_col in df.columns)
                if have_nose:
                    df[enc_col] = self._series_on_food(df[nose_x_col], df[nose_y_col], mask_array)
                else:
                    df[enc_col] = np.nan

                # Stats for panel
                def pct_true(series):
                    vals = pd.Series(series).dropna()
                    if len(vals) == 0:
                        return 0, 0.0
                    on = (vals == 1).sum()
                    return len(vals), (on / len(vals)) * 100.0

                n_cent, p_cent = pct_true(df[cent_col])
                lines.append(f"{wid}: centroid_on_food valid={n_cent}, on_food%={p_cent:.1f}")

                if have_nose:
                    n_enc, p_enc = pct_true(df[enc_col])
                    lines.append(f"{wid}: food_encounter (nose) valid={n_enc}, on_food%={p_enc:.1f}")
                else:
                    lines.append(f"{wid}: food_encounter (nose) not available (no nose cols)")
                lines.append("")

            # Order columns: frame, then per-worm x,y,nose_x,nose_y,centroid_on_food,food_encounter
            ordered = []
            if 'frame' in df.columns:
                ordered.append('frame')
            for wid in self.worm_ids:
                for c in [f"{wid}_x", f"{wid}_y", f"{wid}_nose_x", f"{wid}_nose_y",
                          f"{wid}_centroid_on_food", f"{wid}_food_encounter"]:
                    if c in df.columns:
                        ordered.append(c)

            # Append any remaining columns; drop old on_food columns
            for c in df.columns:
                if c not in ordered:
                    if c.endswith('_on_food_nose') or c.endswith('_on_food'):
                        continue
                    ordered.append(c)

            df = df[ordered]
            self.processed_data = df

            food_txt = "White=ON food" if self.white_is_food.get() else "White=OFF food"
            lines.append(food_txt)
            lines.append(f"Rows: {len(df)}, Objects: {len(self.worm_ids)}")
            self.results_info.insert("end", "\n".join(lines))
            self.results_info.config(state="disabled")
            self.progress_var.set("CSV processed. Save when ready.")

        except Exception as e:
            messagebox.showerror("Error", f"Processing failed:\n{e}")

    def _save_results(self):
        if self.processed_data is None:
            messagebox.showwarning("Warning", "Nothing to save. Run Process first.")
            return
        if self.current_mask is None:
            messagebox.showwarning("Warning", "No mask available to save.")
            return
            
        path = filedialog.asksaveasfilename(defaultextension=".csv",
                                            filetypes=[("CSV", "*.csv"), ("All files", "*.*")])
        if not path:
            return
        try:
            # Save CSV
            self.processed_data.to_csv(path, index=False)
            
            # Save food boundary coordinates with buffer zone
            base_path = Path(path).stem
            boundary_path = Path(path).parent / f"{base_path}_food_boundary.json"
            
            # Use original image size if available
            if self.original_image is not None:
                oh, ow = self.original_image.shape[:2]
                mask_array = cv2.resize(self.current_mask.astype(np.uint8), (ow, oh),
                                       interpolation=cv2.INTER_NEAREST)
            else:
                mask_array = self.current_mask.astype(np.uint8)
            
            # Standardize: 1 = food in the mask_array
            if not self.white_is_food.get():
                # Checkbox says black=food, so invert: make food pixels = 1
                mask_array = 1 - mask_array
            
            # Validate mask before proceeding
            total_pixels = mask_array.shape[0] * mask_array.shape[1]
            food_pixels = np.sum(mask_array)
            food_percentage = (food_pixels / total_pixels) * 100
            
            if food_percentage > 95:
                response = messagebox.askyesno(
                    "Warning: Suspicious Mask",
                    f"Food region is {food_percentage:.1f}% of the image.\n"
                    f"This suggests the mask may be inverted.\n\n"
                    f"Current setting: {'White' if self.white_is_food.get() else 'Black'} = food\n\n"
                    f"Continue anyway?"
                )
                if not response:
                    return
            elif food_percentage < 5:
                response = messagebox.askyesno(
                    "Warning: Small Food Region",
                    f"Food region is only {food_percentage:.1f}% of the image.\n"
                    f"This seems unusually small.\n\n"
                    f"Current setting: {'White' if self.white_is_food.get() else 'Black'} = food\n\n"
                    f"Continue anyway?"
                )
                if not response:
                    return
            
            # Extract contours
            contours, _ = cv2.findContours(mask_array, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if len(contours) == 0:
                messagebox.showwarning("Warning", "No food boundary found in mask")
                return
            
            # Use largest contour (main food patch)
            main_contour = max(contours, key=cv2.contourArea)
            
            # Calculate food metrics first
            area = cv2.contourArea(main_contour)
            perimeter = cv2.arcLength(main_contour, True)
            M = cv2.moments(main_contour)
            if M['m00'] > 0:
                centroid_x = M['m10'] / M['m00']
                centroid_y = M['m01'] / M['m00']
            else:
                centroid_x, centroid_y = 0, 0
            
            # Get EXACT boundary pixels using morphological gradient (1-pixel transitions)
            kernel = np.ones((3, 3), np.uint8)
            gradient = cv2.morphologyEx(mask_array, cv2.MORPH_GRADIENT, kernel)
            
            # Extract boundary pixel coordinates
            boundary_pixels = np.argwhere(gradient > 0)
            boundary_coords = [[int(x), int(y)] for y, x in boundary_pixels]
            
            # Contour points - use exact points from findContours (no simplification)
            contour_points = [[int(pt[0][0]), int(pt[0][1])] for pt in main_contour]
            
            # Build JSON structure
            import json
            boundary_data = {
                "mask_dimensions": {"width": int(mask_array.shape[1]),
                                   "height": int(mask_array.shape[0])},
                "food_metrics": {
                    "area_pixels": float(area),
                    "area_percentage": float((area / total_pixels) * 100),
                    "perimeter_pixels": float(perimeter),
                    "centroid": {"x": float(centroid_x), "y": float(centroid_y)}
                },
                "boundary_region": {
                    "description": "Exact 1-pixel boundary (transition pixels between food and non-food)",
                    "num_pixels": len(boundary_coords),
                    "coordinates": boundary_coords  # [x, y] pairs
                },
                "food_contour": {
                    "description": "Exact food boundary contour points (no simplification)",
                    "num_points": len(contour_points),
                    "coordinates": contour_points  # [x, y] pairs
                }
            }
            
            with open(boundary_path, 'w') as f:
                json.dump(boundary_data, f, indent=2)
            
            self.progress_var.set(f"Saved CSV + boundary data")
            messagebox.showinfo("Saved", 
                f"Results saved:\n{path}\n\nBoundary data saved:\n{boundary_path}\n\n"
                f"Food area: {area:.0f} px² ({(area/total_pixels)*100:.1f}%)\n"
                f"Contour points: {len(contour_points)} | Boundary pixels: {len(boundary_coords)}")
        except Exception as e:
            import traceback
            messagebox.showerror("Error", f"Save failed:\n{e}\n\n{traceback.format_exc()}")

def main():
    root = tk.Tk()
    app = AutoMaskNoseApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
