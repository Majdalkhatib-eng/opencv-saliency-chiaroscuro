import cv2
import numpy as np

class ChiaroscuroNormalizer:
    """
    Adaptive Tone-Mapping & Micro-Contrast Enhancement.
    Implements Highlight Knee Compression, Logarithmic Gamma Adaptation,
    and Spatial High-Pass Unsharp Masking.
    """
    def __init__(self, target_luma=0.25, max_p95=0.68, shadow_crush=1.08):
        self.target_luma = target_luma
        self.max_p95 = max_p95
        self.shadow_crush = shadow_crush

    def process(self, bgr_img):
        img_f = bgr_img.astype(np.float32) / 255.0

        # 1. حساب النصوع القياسي ITU-R BT.601
        luma = 0.114 * img_f[:, :, 0] + 0.587 * img_f[:, :, 1] + 0.299 * img_f[:, :, 2]
        p95_l = float(np.percentile(luma, 95))

        # 2. ضغط المناطق شديدة السطوع لمنع الاحتراق
        if p95_l > self.max_p95:
            excess = p95_l - self.max_p95
            knee = 0.45
            high_mask = np.clip((luma - knee) / (1.0 - knee + 1e-5), 0.0, 1.0)[:, :, np.newaxis]
            pull_factor = max(0.55, 1.0 - excess * 1.10)
            img_f = img_f * (1.0 - high_mask * (1.0 - pull_factor))

        # 3. تصحيح غاما لوغاريتمي تكيفي لموازنة المشهد
        luma_adj = 0.114 * img_f[:, :, 0] + 0.587 * img_f[:, :, 1] + 0.299 * img_f[:, :, 2]
        cur_mean = float(np.mean(luma_adj))
        if cur_mean > self.target_luma * 1.05:
            gamma = np.clip(np.log(self.target_luma) / np.log(max(0.05, cur_mean)), 1.05, 1.70)
            img_f = np.power(np.clip(img_f, 0.0, 1.0), gamma)
        elif cur_mean < self.target_luma * 0.70:
            gamma = np.clip(np.log(self.target_luma * 0.85) / np.log(max(0.02, cur_mean)), 0.80, 0.96)
            img_f = np.power(np.clip(img_f, 0.0, 1.0), gamma)

        img_f = np.power(np.clip(img_f, 0.0, 1.0), self.shadow_crush)

        # 4. تعزيز التفاصيل الدقيقة (Micro-Detail Unsharp Masking) في الدرجات المتوسطة
        luma_final = 0.114 * img_f[:, :, 0] + 0.587 * img_f[:, :, 1] + 0.299 * img_f[:, :, 2]
        blur_local = cv2.GaussianBlur(luma_final, (0, 0), sigmaX=9.0)
        micro_detail = (luma_final - blur_local)[:, :, np.newaxis]
        midtone_mask = (4.0 * luma_final * (1.0 - luma_final))[:, :, np.newaxis]

        img_f = np.clip(img_f + micro_detail * midtone_mask * 0.14, 0.0, 1.0)
        return np.ascontiguousarray((img_f * 255.0).astype(np.uint8))