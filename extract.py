#!/usr/bin/env python3
"""
Manual plot digitizer for PDF plots.

Workflow:
1) Render the PDF page.
2) Optionally crop the plot area.
3) Click calibration points on the axes.
4) Click points on each curve, one curve at a time.
5) Export each curve to CSV.

Controls:
- Mouse wheel: zoom in/out around cursor
- Left click: add a point
- u: undo last point
- r: reset zoom
- Enter: finish current curve
- Esc: cancel current curve

Install:
  pip install pymupdf matplotlib numpy pandas opencv-python
"""

import os
import fitz
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


PDF_PATH = "./datasheets/4CW150000E.pdf"
PAGE_INDEX = 8   # page 9 in the PDF, zero-based
DPI = 350

# Axis values for the page 9 plot
X0_VAL = 1     # UA = 0 kV
X1_VAL = 20    # UA = 12 kV
Y0_VAL = -600   # UG1 at bottom of plot
Y1_VAL = 800    # UG1 at top of plot


def render_pdf_page(pdf_path: str, page_index: int, dpi: int = 350) -> np.ndarray:
    doc = fitz.open(pdf_path)
    page = doc[page_index]
    mat = fitz.Matrix(dpi / 72.0, dpi / 72.0)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if pix.n == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    return img


class ImagePicker:
    def __init__(self, img: np.ndarray, title: str):
        self.img = img
        self.title = title
        self.points = []
        self.finished = False
        self.cancelled = False

        self.fig, self.ax = plt.subplots(figsize=(12, 16))
        self.ax.imshow(self.img)
        self.ax.set_title(
            title + "\n"
            "Mouse wheel = zoom, left click = add point, u = undo, r = reset, Enter = finish"
        )
        self.ax.axis("off")

        self.orig_xlim = self.ax.get_xlim()
        self.orig_ylim = self.ax.get_ylim()

        self.cid_scroll = self.fig.canvas.mpl_connect("scroll_event", self.on_scroll)
        self.cid_click = self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.cid_key = self.fig.canvas.mpl_connect("key_press_event", self.on_key)

    def on_scroll(self, event):
        if event.xdata is None or event.ydata is None:
            return

        scale = 1.2
        if event.button == "up":
            factor = 1 / scale
        elif event.button == "down":
            factor = scale
        else:
            return

        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()

        xdata = event.xdata
        ydata = event.ydata

        new_xmin = xdata - (xdata - xlim[0]) * factor
        new_xmax = xdata + (xlim[1] - xdata) * factor
        new_ymin = ydata - (ydata - ylim[0]) * factor
        new_ymax = ydata + (ylim[1] - ydata) * factor

        self.ax.set_xlim(new_xmin, new_xmax)
        self.ax.set_ylim(new_ymin, new_ymax)
        self.fig.canvas.draw_idle()

    def on_click(self, event):
        if event.inaxes != self.ax:
            return
        if event.button != 1:
            return
        if event.xdata is None or event.ydata is None:
            return

        self.points.append((event.xdata, event.ydata))
        idx = len(self.points)

        self.ax.plot(event.xdata, event.ydata, "ro", markersize=4)
        self.ax.text(event.xdata + 4, event.ydata + 4, str(idx), color="red", fontsize=9)
        self.fig.canvas.draw_idle()

    def on_key(self, event):
        if event.key == "enter":
            self.finished = True
            plt.close(self.fig)
        elif event.key == "escape":
            self.cancelled = True
            plt.close(self.fig)
        elif event.key == "u":
            if self.points:
                self.points.pop()
                self.redraw()
        elif event.key == "r":
            self.ax.set_xlim(self.orig_xlim)
            self.ax.set_ylim(self.orig_ylim)
            self.fig.canvas.draw_idle()

    def redraw(self):
        self.ax.clear()
        self.ax.imshow(self.img)
        self.ax.set_title(
            self.title + "\n"
            "Mouse wheel = zoom, left click = add point, u = undo, r = reset, Enter = finish"
        )
        self.ax.axis("off")

        for i, (x, y) in enumerate(self.points, start=1):
            self.ax.plot(x, y, "ro", markersize=4)
            self.ax.text(x + 4, y + 4, str(i), color="red", fontsize=9)

        self.fig.canvas.draw_idle()

    def show(self):
        plt.show()
        if self.cancelled:
            return None
        return self.points


def crop_image(img: np.ndarray):
    print("Click two opposite corners of the plot area to crop it.")
    picker = ImagePicker(img, "Crop plot area")
    pts = picker.show()
    if pts is None or len(pts) != 2:
        raise RuntimeError("Crop cancelled or incomplete.")

    (x1, y1), (x2, y2) = pts
    left = int(round(min(x1, x2)))
    right = int(round(max(x1, x2)))
    top = int(round(min(y1, y2)))
    bottom = int(round(max(y1, y2)))

    return img[top:bottom, left:right].copy(), (left, top)


def pick_calibration_points(img: np.ndarray):
    print("Click calibration points in this order:")
    print(f"  1) x-axis point at UA = {X0_VAL} kV")
    print(f"  2) x-axis point at UA = {X1_VAL} kV")
    print(f"  3) y-axis point at UG1 = {Y0_VAL} V")
    print(f"  4) y-axis point at UG1 = {Y1_VAL} V")

    picker = ImagePicker(img, "Calibration points")
    pts = picker.show()
    if pts is None or len(pts) != 4:
        raise RuntimeError("Calibration cancelled or incomplete.")
    return pts


def make_axis_maps(calib_pts):
    x0p, _ = calib_pts[0]
    x1p, _ = calib_pts[1]
    _, y0p = calib_pts[2]
    _, y1p = calib_pts[3]

    def x_to_val(x_pix):
        return X0_VAL + (x_pix - x0p) * (X1_VAL - X0_VAL) / (x1p - x0p)

    def y_to_val(y_pix):
        return Y0_VAL + (y_pix - y0p) * (Y1_VAL - Y0_VAL) / (y1p - y0p)

    return x_to_val, y_to_val


def digitize_curve(img: np.ndarray, x_to_val, y_to_val, curve_name: str):
    print(f"\nDigitizing curve: {curve_name}")
    print("Click points along the curve in any order you like, then press Enter.")
    picker = ImagePicker(img, f"Curve: {curve_name}")
    pts = picker.show()
    if pts is None or len(pts) < 2:
        raise RuntimeError("Curve digitization cancelled or too few points.")

    rows = []
    for x_pix, y_pix in pts:
        rows.append({
            "x_pix": x_pix,
            "y_pix": y_pix,
            "UA_kV": x_to_val(x_pix),
            "UG1_V": y_to_val(y_pix),
        })

    df = pd.DataFrame(rows)
    df = df.sort_values("UA_kV").reset_index(drop=True)
    return df


def main():
    if not os.path.exists(PDF_PATH):
        raise FileNotFoundError(PDF_PATH)

    img = render_pdf_page(PDF_PATH, PAGE_INDEX, DPI)

    crop, crop_origin = crop_image(img)
    calib_pts = pick_calibration_points(crop)
    x_to_val, y_to_val = make_axis_maps(calib_pts)

    out_dir = "manual_digitized_curves"
    os.makedirs(out_dir, exist_ok=True)

    curve_idx = 1
    while True:
        curve_name = input(
            f"\nName for curve {curve_idx} "
            f"(or press Enter to stop): "
        ).strip()

        if curve_name == "":
            break

        df = digitize_curve(crop, x_to_val, y_to_val, curve_name)
        csv_path = os.path.join(out_dir, f"{curve_idx:02d}_{curve_name}.csv")
        df.to_csv(csv_path, index=False)
        print(f"Saved {csv_path} with {len(df)} points.")
        curve_idx += 1

    print(f"\nDone. Files saved in: {out_dir}")


if __name__ == "__main__":
    main()
