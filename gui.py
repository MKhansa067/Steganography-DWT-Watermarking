import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import cv2, os, numpy as np

from watermark_dwt import embed_watermark, extract_watermark, correlation, _text_to_image
from watermark_dwt import _calc_psnr
from utils import add_salt_pepper, median_filter, Timer

DEFAULT_ALPHA = 0.4
DEFAULT_LEVEL = 1
CANVAS_SIZE   = (220, 220)
NOISE_LIMITS  = (0.01, 0.30)

class WatermarkApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Image Watermarking using DWT")
        self.geometry("1000x500")
        self.resizable(False, False)
        self.last_loaded_image = None

        self.cover_path = self.wm_source = self.orig_wm_img = None
        self.stego_path = self.noisy_path = self.filtered_path = None

        self._build_widgets()

    def _build_widgets(self):
        menu = tk.Frame(self, bd=2, relief="groove")
        menu.place(x=10, y=10, width=150, height=480)

        ttk.Button(menu, text="Select Cover Image",    command=self.select_cover     ).pack(fill="x", pady=4)
        ttk.Button(menu, text="Select Watermark",      command=self.select_watermark ).pack(fill="x", pady=4)
        ttk.Button(menu, text="Embedding",             command=self.do_embedding     ).pack(fill="x", pady=4)
        ttk.Button(menu, text="Download Stego Image",  command=self.download_stego   ).pack(fill="x", pady=4)
        ttk.Button(menu, text="Retrival",              command=self.show_retrieval   ).pack(fill="x", pady=4)
        ttk.Button(menu, text="Clear All",             command=self.clear_all        ).pack(fill="x", pady=4)
        ttk.Button(menu, text="Exit",                  command=self.destroy          ).pack(side="bottom", fill="x", pady=4)

        self.status_var = tk.StringVar(value="Ready")
        tk.Label(self, textvariable=self.status_var, bd=2, relief="sunken",
                 anchor="w").place(x=170, y=10, width=820, height=30)

        noise = tk.LabelFrame(self, text="Noise Attack")
        noise.place(x=170, y=50, width=820, height=80)

        tk.Label(noise, text="Enter Noise Amount (0.01-0.3):").place(x=10, y=10)
        self.noise_var = tk.DoubleVar(value=0.1)
        tk.Entry(noise, textvariable=self.noise_var, width=6).place(x=190, y=10)

        ttk.Button(noise, text="Select Watermarked Image", command=self.select_stego).place(x=260, y=5)
        ttk.Button(noise, text="Add Noise",  command=self.add_noise   ).place(x=440, y=5)
        ttk.Button(noise, text="Filter",     command=self.apply_filter).place(x=530, y=5)
        ttk.Button(noise, text="Extract",    command=self.extract_from_noisy).place(x=600, y=5)

        disp = tk.LabelFrame(self, text="Image Display Bar")
        disp.place(x=170, y=140, width=820, height=220)

        self.canvas1 = self._add_canvas(disp,  20)
        self.canvas2 = self._add_canvas(disp, 220)
        self.canvas3 = self._add_canvas(disp, 420)
        self.canvas4 = self._add_canvas(disp, 620)

        tk.Label(disp, text="Cover / Stego").place(x=20,  y=185)
        tk.Label(disp, text="Watermark / Noisy").place(x=220, y=185)
        tk.Label(disp, text="Stego / Filtered").place(x=420, y=185)
        tk.Label(disp, text="Recovered Watermark").place(x=620, y=185)

        out = tk.LabelFrame(self, text="Output Parameters")
        out.place(x=170, y=370, width=820, height=120)

        tk.Label(out, text="PSNR (dB):").place(x=20, y=10)
        self.psnr_var = tk.StringVar(); tk.Label(out, textvariable=self.psnr_var,
                 width=12, relief="sunken").place(x=100, y=10)

        tk.Label(out, text="Correlation factor:").place(x=250, y=10)
        self.corr_var = tk.StringVar(); tk.Label(out, textvariable=self.corr_var,
                 width=12, relief="sunken").place(x=380, y=10)

        tk.Label(out, text="Total Elapsed Time (s):").place(x=520, y=10)
        self.time_var = tk.StringVar(); tk.Label(out, textvariable=self.time_var,
                 width=12, relief="sunken").place(x=690, y=10)

    def _add_canvas(self, parent, x):
        c = tk.Label(parent, bd=1, relief="solid", width=CANVAS_SIZE[0], height=CANVAS_SIZE[1])
        c.place(x=x, y=10)
        return c

    def _display_array(self, canvas, arr):
        img = Image.fromarray(arr)
        img = img.resize(CANVAS_SIZE, Image.Resampling.NEAREST)
        tk_img = ImageTk.PhotoImage(img)
        canvas.image = tk_img
        canvas.config(image=tk_img)

    def select_cover(self):
        path = filedialog.askopenfilename(filetypes=[("Image", "*.png;*.jpg;*.jpeg;*.bmp")])
        if path:
            self.cover_path = path
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            self._display_array(self.canvas1, img)
            self.status_var.set(f"Cover image selected: {os.path.basename(path)}")

    def select_watermark(self):
        if messagebox.askyesno("Watermark", "Gunakan gambar?\nPilih No untuk mengetik teks."):
            path = filedialog.askopenfilename(filetypes=[("Image", "*.png;*.jpg;*.jpeg;*.bmp")])
            if path:
                self.wm_source   = path
                self.orig_wm_img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        else:
            text = simpleinput(self, "Input Text Watermark", "Masukkan teks watermark:")
            if not text: return
            self.wm_source   = text
            self.orig_wm_img = _text_to_image(text)

        self._display_array(self.canvas2, self.orig_wm_img)
        self.status_var.set("Watermark siap.")

    def do_embedding(self):
        if not (self.cover_path and self.wm_source):
            messagebox.showerror("Error", "Pilih cover dan watermark dahulu.")
            return
        with Timer() as t:
            stego_img, psnr = embed_watermark(self.cover_path, self.wm_source,
                                              "temp_stego.png",
                                              alpha=DEFAULT_ALPHA, level=DEFAULT_LEVEL)
        self.stego_path = "temp_stego.png"
        self.psnr_var.set(f"{psnr:.2f}")
        self.time_var.set(f"{t.elapsed:.3f}")
        self._display_array(self.canvas3, stego_img)
        self.status_var.set("Embedding selesai – watermarked image di-canvas 3.")

    def download_stego(self):
        if not self.stego_path:
            messagebox.showerror("Error", "Belum ada stego image untuk disimpan.")
            return
        save_path = filedialog.asksaveasfilename(defaultextension=".png",
                                                 filetypes=[("PNG Image", "*.png")],
                                                 title="Save Stego Image As")
        if save_path:
            img = cv2.imread(self.stego_path)
            cv2.imwrite(save_path, img)
            messagebox.showinfo("Success", f"Stego image berhasil disimpan:\n{save_path}")

    def show_retrieval(self):
        if not self.stego_path:
            messagebox.showerror("Error", "Belum ada stego image.")
            return
        rec = extract_watermark(self.stego_path, alpha=DEFAULT_ALPHA, level=DEFAULT_LEVEL)

        if self.orig_wm_img is not None:
            wm_ref = cv2.resize(self.orig_wm_img, (rec.shape[1], rec.shape[0]),
                                interpolation=cv2.INTER_NEAREST)
            corr = correlation(wm_ref, rec)
        else:
            corr = 0.0

        self.corr_var.set(f"{corr:.3f}")
        self._display_array(self.canvas4, rec)
        self.status_var.set("Watermark berhasil diekstrak (canvas 4).")

    def select_stego(self):
        path = filedialog.askopenfilename(filetypes=[("Image", "*.png;*.jpg;*.jpeg;*.bmp")])
        if path:
            self.stego_path = path
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            self._display_array(self.canvas1, img)
            self.last_loaded_image = "stego"
            self.status_var.set(f"Stego image loaded: {os.path.basename(path)}")

    def add_noise(self):
        if not self.stego_path:
            messagebox.showerror("Error", "Load stego image dahulu.")
            return
        amount = self.noise_var.get()
        if not (NOISE_LIMITS[0] <= amount <= NOISE_LIMITS[1]):
            messagebox.showerror("Error", "Noise amount di luar rentang.")
            return
        img = cv2.imread(self.stego_path, cv2.IMREAD_GRAYSCALE)
        noisy = add_salt_pepper(img, amount)
        cv2.imwrite("temp_noisy.png", noisy)
        self.noisy_path = "temp_noisy.png"
        self._display_array(self.canvas2, noisy)
        self.last_loaded_image = "noisy"
        self.status_var.set(f"Noise {amount} ditambahkan (canvas 2).")

    def apply_filter(self):
        if not self.noisy_path:
            messagebox.showerror("Error", "Tidak ada citra noisy.")
            return
        noisy = cv2.imread(self.noisy_path, cv2.IMREAD_GRAYSCALE)
        filt = median_filter(noisy, 3)
        cv2.imwrite("temp_filtered.png", filt)
        self.filtered_path = "temp_filtered.png"
        self._display_array(self.canvas3, filt)
        self.last_loaded_image = "filtered"
        self.status_var.set("Median filter selesai (canvas 3).")

    def extract_from_noisy(self):
        if self.last_loaded_image == "filtered" and self.filtered_path:
            path = self.filtered_path
        elif self.last_loaded_image == "noisy" and self.noisy_path:
            path = self.noisy_path
        elif self.last_loaded_image == "stego" and self.stego_path:
            path = self.stego_path
        else:
            messagebox.showerror("Error", "Tidak ada citra untuk ekstraksi watermark.")
            return
        rec = extract_watermark(path, alpha=DEFAULT_ALPHA, level=DEFAULT_LEVEL)
        if self.orig_wm_img is not None:
            wm_ref = cv2.resize(self.orig_wm_img,
                                (rec.shape[1], rec.shape[0]),
                                interpolation=cv2.INTER_NEAREST)
            corr = correlation(wm_ref, rec)
        else:
            corr = 0
        self.corr_var.set(f"{corr:.3f}")
        self._display_array(self.canvas4, rec)
        self.status_var.set(f"Ekstraksi watermark dari {self.last_loaded_image} image selesai.")

    def clear_all(self):
        self.cover_path = self.wm_source = self.stego_path = None
        self.noisy_path = self.filtered_path = None
        self.orig_wm_img = None
        self.psnr_var.set(""); self.corr_var.set(""); self.time_var.set("")
        for c in [self.canvas1, self.canvas2, self.canvas3, self.canvas4]:
            c.config(image=""); c.image = None
        self.status_var.set("Semua data dibersihkan.")

def simpleinput(root, title, prompt):
    win = tk.Toplevel(root); win.title(title); win.grab_set()
    tk.Label(win, text=prompt).pack(padx=10, pady=6)
    var = tk.StringVar(); entry = tk.Entry(win, textvariable=var, width=30)
    entry.pack(padx=10, pady=6); entry.focus()
    data = {"val": None}
    ttk.Button(win, text="OK", command=lambda: (data.update(val=var.get()), win.destroy())).pack(pady=6)
    root.wait_window(win); return data["val"]

if __name__ == "__main__":
    WatermarkApp().mainloop()