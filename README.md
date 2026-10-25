# OpenCV Saliency Smart Crop & Chiaroscuro Normalizer

A lightweight, dependency-minimal **Computer Vision** library in Python and OpenCV designed for content-aware intelligent image cropping and adaptive dynamic-range normalization.

---

## Technical Highlights

* **Sobel-Based Saliency Detection:** Computes spatial directional gradients ($G_x, G_y$) combined with midtone luminosity weighting and a 2D Gaussian spatial prior to locate focal subjects without requiring heavy deep-learning inference.
* **Aspect Ratio Re-Framing:** Dynamically reframes landscape images (16:9) into vertical cinema formats (9:16) while keeping focal points securely framed.
* **Highlight Knee Compression:** Analyzes the 95th percentile luminance to suppress overexposed regions and protect textural integrity.
* **Logarithmic Gamma Tone-Mapping:** Automatically computes adaptive gamma parameters $\gamma = \frac{\ln(L_{\text{target}})}{\ln(L_{\text{mean}})}$ to balance illumination across heterogeneous assets.
* **Micro-Contrast Frequency Extraction:** Employs spatial high-pass unsharp masking gated by parabolic midtone masks $4L(1-L)$ to accentuate fine textures without amplifying shadow noise.

---

## Mathematical Details

### 1. Saliency Energy Formulation
$$\text{Saliency}(x, y) = \|\nabla I(x, y)\|^{1.2} \cdot [4.2 I (1 - I)] \cdot \exp\left(-\left[\frac{(x - \mu_x)^2}{2\sigma_x^2} + \frac{(y - \mu_y)^2}{2\sigma_y^2}\right]\right)$$

### 2. Adaptive Gamma Tone-Mapping
$$\gamma = \text{clip}\left(\frac{\ln(L_{\text{target}})}{\ln(\max(0.05, \bar{L}))}, \gamma_{\min}, \gamma_{\max}\right)$$
$$I_{\text{out}} = I_{\text{in}}^\gamma$$

---

## Installation

```bash
git clone [https://github.com/YOUR_USERNAME/opencv-saliency-chiaroscuro.git](https://github.com/YOUR_USERNAME/opencv-saliency-chiaroscuro.git)
cd opencv-saliency-chiaroscuro
pip install -r requirements.txt
```

---

## Quickstart

Run the demonstration script:

```bash
python demo.py
```

---

## License

This project is licensed under the MIT License.