import cv2
import numpy as np
from src.saliency_cropper import ContentAwareCropper
from src.chiaroscuro_filter import ChiaroscuroNormalizer

def main():
    # توليد صورة اختبارية تحتوي على دائرة غير متمركزة لإثبات القص الذكي
    w, h = 1920, 1080
    test_img = np.zeros((h, w, 3), dtype=np.uint8)
    # إضافة خلفية متدرجة وجسم ساطع على الطرف
    for y in range(h):
        test_img[y, :] = int(40 * (y / h))
    cv2.circle(test_img, (1350, 480), 220, (220, 210, 195), -1)
    cv2.putText(test_img, "FOCAL SUBJECT", (1220, 490), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (20, 20, 20), 2)

    cv2.imwrite("input_landscape_raw.png", test_img)
    print("[*] Saved raw test image: input_landscape_raw.png")

    # 1. اختبار كاشف البروز والقص العمودي (Smart Crop: 9:16)
    print("[*] Detecting saliency center and cropping to 9:16 (1080x1920)...")
    cropper = ContentAwareCropper(target_w=1080, target_h=1920)
    cx, cy = cropper.find_subject_center(test_img)
    print(f"    [+] Detected Saliency Focal Center: (X: {cx:.2f}, Y: {cy:.2f})")
    cropped_vertical = cropper.crop_and_scale(test_img)
    cv2.imwrite("output_smart_cropped.png", cropped_vertical)

    # 2. تطبيق معالج الإضاءة الكلاسيكية والـ Micro-Contrast
    print("[*] Applying Chiaroscuro Dynamic Range Normalization...")
    normalizer = ChiaroscuroNormalizer(target_luma=0.25)
    enhanced = normalizer.process(cropped_vertical)
    cv2.imwrite("output_chiaroscuro_enhanced.png", enhanced)

    print("[OK] Processing complete! Check output_smart_cropped.png and output_chiaroscuro_enhanced.png")

if __name__ == "__main__":
    main()