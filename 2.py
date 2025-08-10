import numpy as np
import cv2
from tensorflow.keras.applications import ResNet101
from tensorflow.keras.models import Model
from tensorflow.keras.applications.resnet import preprocess_input
import matplotlib.pyplot as plt

# =========================================================
# 1. Approximate Adaptive Median Filter (fast)
# =========================================================
def fast_adaptive_median(img):
    # Mimics adaptive median by progressively applying medianBlur with increasing kernel size
    filtered = img.copy()
    for k in [3, 5, 7, 9]:
        filtered = cv2.medianBlur(filtered, k)
    return filtered

# =========================================================
# 2. Load and preprocess image
# =========================================================
img_path = "../training_set/training_set/cats/cat.41.jpg"
img = cv2.imread(img_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Apply approximate adaptive median filter
smoothed = fast_adaptive_median(img_rgb)

# =========================================================
# 3. Weighted sum transformation (Eq. 1)
# =========================================================
R, G, B = cv2.split(smoothed)
Tpix = 2*R.astype(np.float32) + 3*G.astype(np.float32) + 4*B.astype(np.float32)

# =========================================================
# 4. Directional color differences (vectorized)
# =========================================================
diff0   = np.abs(np.roll(Tpix, -1, axis=1) - np.roll(Tpix, 1, axis=1))     # horizontal
diff90  = np.abs(np.roll(Tpix, -1, axis=0) - np.roll(Tpix, 1, axis=0))     # vertical
diff135 = np.abs(np.roll(np.roll(Tpix, -1, axis=0), -1, axis=1) -
                 np.roll(np.roll(Tpix, 1, axis=0), 1, axis=1))             # 135°
diff45  = np.abs(np.roll(np.roll(Tpix, 1, axis=0), -1, axis=1) -
                 np.roll(np.roll(Tpix, -1, axis=0), 1, axis=1))            # 45°

# Zero out invalid border values
diff0[:, 0] = diff0[:, -1] = 0
diff90[0, :] = diff90[-1, :] = 0
diff135[0, :] = diff135[-1, :] = 0
diff45[0, :] = diff45[-1, :] = 0

# =========================================================
# 5. Max directional difference
# =========================================================
max_diff = np.maximum.reduce([diff0, diff90, diff135, diff45])

# =========================================================
# 6. Thresholding
# =========================================================
T = 1.2 * np.mean(max_diff)
edge_map = (max_diff >= T).astype(np.uint8) * 255

# =========================================================
# 7. Two-mask thinning (fast morphological)
# =========================================================
def thinning_two_masks(edge_img):
    # Structuring elements for horizontal and vertical thinning
    mask_h = cv2.getStructuringElement(cv2.MORPH_RECT, (3,1))
    mask_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1,3))
    # Erode and keep only thinnest lines
    thin_h = cv2.erode(edge_img, mask_h)
    thin_v = cv2.erode(edge_img, mask_v)
    return cv2.max(thin_h, thin_v)

thinned_edges = thinning_two_masks(edge_map)

# =========================================================
# 7b. Invert for white background and lighten edges
# =========================================================
# Invert the edge map to make background white, edges black
edges_inverted = cv2.bitwise_not(thinned_edges)

# Lighten edges to soft gray instead of full black
edges_light = cv2.addWeighted(edges_inverted, 0.8, 255*np.ones_like(edges_inverted), 0.2, 0)

# =========================================================
# 8. Pass edges to ResNet101 layer 1
# =========================================================
edges_rgb = cv2.merge([thinned_edges]*3)
edges_resized = cv2.resize(edges_rgb, (224, 224))
x = np.expand_dims(edges_resized.astype(np.float32), axis=0)
x = preprocess_input(x)

base_model = ResNet101(weights='imagenet', include_top=False)
intermediate_model = Model(inputs=base_model.input, outputs=base_model.layers[1].output)
feature_map = intermediate_model.predict(x)[0]

channel_idx = 1
single_channel = feature_map[:, :, channel_idx]
# Normalize as before
norm_channel = cv2.normalize(single_channel, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# Invert for white background
norm_channel_inverted = cv2.bitwise_not(norm_channel)

# Lighten edges (0.8 weight keeps edges, 0.2 pushes toward white)
norm_channel_light = cv2.addWeighted(norm_channel_inverted, 0.8, 255*np.ones_like(norm_channel_inverted), 0.2, 0)

# =========================================================
# 9. Display results
# =========================================================
plt.figure(figsize=(15,5))
plt.subplot(1,3,1)
plt.imshow(img_rgb)
plt.title("Original RGB Image")

plt.subplot(1,3,2)
plt.imshow(edges_light, cmap='gray')
plt.title("Final Edge Map (Fast Paper-like)")

plt.subplot(1,3,3)
plt.imshow(norm_channel_light, cmap='gray')
plt.title("ResNet Layer 1 - Single Channel")
plt.show()
