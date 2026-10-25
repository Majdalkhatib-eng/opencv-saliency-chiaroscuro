import cv2
import numpy as np

class ContentAwareCropper:
    """
    Lightweight Computer Vision Saliency Detector using Directional Sobel Derivatives,
    Midtone Weighting, and a 2D Gaussian Spatial Prior.
    """
    def __init__(self, target_w=1080, target_h=1920):
        self.target_w = target_w
        self.target_h = target_h

    def find_subject_center(self, bgr_img):
        """
        Computes focal coordinates (cx, cy) normalized between [0, 1].
        """
        small = cv2.resize(bgr_img, (320, 180), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

        # 1. مشتقات سوبل الاتجاهية لحساب شدة الحواف
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        edge_mag = np.sqrt(gx**2 + gy**2)

        # 2. قناع وزن الدرجات المتوسطة لتقليل ضجيج السطوع الشديد
        midtone_weight = np.clip(4.2 * gray * (1.0 - gray), 0.0, 1.0)

        # 3. توزيع احتمالي غاوسي ثنائي الأبعاد يفضل العناصر القريبة من المركز
        sh, sw = gray.shape
        yy, xx = np.ogrid[:sh, :sw]
        spatial_prior = np.exp(-(((xx - sw * 0.5) / (sw * 0.42))**2 + ((yy - sh * 0.42) / (sh * 0.38))**2))

        # 4. دمج مصفوفة البروز وتنعيمها
        saliency = (edge_mag ** 1.2) * midtone_weight * spatial_prior
        saliency = cv2.GaussianBlur(saliency, (21, 21), 0)

        _, max_val, _, max_loc = cv2.minMaxLoc(saliency)
        if max_val > 1e-4:
            cx = np.clip(max_loc[0] / float(sw), 0.25, 0.75)
            cy = np.clip(max_loc[1] / float(sh), 0.25, 0.75)
        else:
            cx, cy = 0.50, 0.45
        return float(cx), float(cy)

    def crop_and_scale(self, bgr_img):
        vh, vw = bgr_img.shape[:2]
        target_ratio = self.target_w / float(self.target_h)
        current_ratio = vw / float(vh)

        if abs(current_ratio - target_ratio) < 0.02:
            return cv2.resize(bgr_img, (self.target_w, self.target_h), interpolation=cv2.INTER_LANCZOS4)

        cx, cy = self.find_subject_center(bgr_img)

        if current_ratio > target_ratio:
            new_w = max(1, int(vh * target_ratio))
            offset_x = int(np.clip(cx * vw - new_w * 0.5, 0, vw - new_w))
            cropped = bgr_img[:, offset_x:offset_x + new_w]
        else:
            new_h = max(1, int(vw / target_ratio))
            offset_y = int(np.clip(cy * vh - new_h * 0.5, 0, vh - new_h))
            cropped = bgr_img[offset_y:offset_y + new_h, :]

        return cv2.resize(cropped, (self.target_w, self.target_h), interpolation=cv2.INTER_LANCZOS4)