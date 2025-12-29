import cv2
import numpy as np
from PIL import Image
import os
from typing import Union, Optional

class FaceCropper:
    """
    Wrapper class cho việc cắt khuôn mặt.
    Hiện tại sử dụng: Haar Cascade (Legacy method).
    """

    def __init__(
        self,
        out_size: int = 256,
        target_face_ratio: float = 1.20,
        scale_factor: float = 1.1,
        min_neighbors: int = 4
    ):
        self.out_size = out_size
        self.target_face_ratio = target_face_ratio
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors

        # Load model
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        
    def __call__(self, image_source: Union[str, np.ndarray, Image.Image]) -> Image.Image:
        """
        Hàm chính thực hiện crop.
        
        Args:
            image_source: Có thể là đường dẫn ảnh (str), Numpy array (cv2), hoặc PIL Image.
            
        Returns:
            PIL.Image: Ảnh khuôn mặt đã crop (RGB).
                       Nếu không tìm thấy mặt, trả về ảnh gốc (resize về out_size).
        """
        
        # 1. Chuẩn hóa đầu vào thành Numpy Array (BGR cho OpenCV)
        if isinstance(image_source, str):
            if not os.path.exists(image_source):
                raise FileNotFoundError(f"Image not found: {image_source}")
            image_np = cv2.imread(image_source) # BGR
        elif isinstance(image_source, Image.Image):
            image_np = np.array(image_source.convert('RGB')) 
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR) # RGB -> BGR
        elif isinstance(image_source, np.ndarray):
            image_np = image_source
        else:
            raise ValueError("Unsupported image type")

        if image_np is None:
            raise ValueError("Could not decode image")

        # 2. Logic Detect & Crop (Legacy Haar Cascade)
        # ---------------------------------------------
        gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=(40, 40)
        )

        if len(faces) > 0:
            # Lấy mặt lớn nhất
            x, y, w, h = max(faces, key=lambda b: b[2] * b[3])

            # Tính toán Zoom & Crop
            face_size = max(w, h)
            desired_face_size = self.target_face_ratio * self.out_size
            zoom_factor = desired_face_size / face_size

            # Resize toàn ảnh
            zoomed_image = cv2.resize(
                image_np, None, 
                fx=zoom_factor, fy=zoom_factor, 
                interpolation=cv2.INTER_CUBIC
            )

            # Tính tọa độ mới
            center_x = int((x + w//2) * zoom_factor)
            center_y = int((y + h//2) * zoom_factor)
            
            half = self.out_size // 2
            x1 = center_x - half
            y1 = center_y - half
            x2 = x1 + self.out_size
            y2 = y1 + self.out_size

            # Padding/Clamping đơn giản (Cắt trong vùng ảnh)
            h_z, w_z = zoomed_image.shape[:2]
            x1 = max(0, x1); y1 = max(0, y1)
            x2 = min(w_z, x2); y2 = min(h_z, y2)

            cropped = zoomed_image[y1:y2, x1:x2]
            
            # Resize về đúng chuẩn nếu bị lệch do biên
            if cropped.shape[0] != self.out_size or cropped.shape[1] != self.out_size:
                cropped = cv2.resize(cropped, (self.out_size, self.out_size), interpolation=cv2.INTER_CUBIC)
                
            final_bgr = cropped
            print(f"[FaceCropper] Face detected. Cropped size: {final_bgr.shape}")
        else:
            # Fallback: Không thấy mặt thì resize ảnh gốc về out_size
            print("[FaceCropper] No face detected. Using full image.")
            final_bgr = cv2.resize(image_np, (self.out_size, self.out_size))

        # 3. Chuẩn hóa đầu ra thành PIL Image (RGB)
        final_rgb = cv2.cvtColor(final_bgr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(final_rgb)