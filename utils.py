import cv2
import numpy as np
import time

def add_salt_pepper(image, amount=0.005):
    noisy = image.copy()
    row, col = image.shape
    num_salt = np.ceil(amount * row * col * 0.5).astype(int)
    num_pepper = np.ceil(amount * row * col * 0.5).astype(int)

    coords = (np.random.randint(0, row, num_salt), np.random.randint(0, col, num_salt))
    noisy[coords] = 255
    coords = (np.random.randint(0, row, num_pepper), np.random.randint(0, col, num_pepper))
    noisy[coords] = 0

    return noisy

def median_filter(img: np.ndarray, ksize: int = 3) -> np.ndarray:
    return cv2.medianBlur(img, ksize)

class Timer:
    """Stop-watch sederhana untuk mengukur elapsed-time."""
    def __enter__(self):
        self.t0 = time.perf_counter()
        return self
    def __exit__(self, *args):
        self.t1 = time.perf_counter()
        self.elapsed = self.t1 - self.t0