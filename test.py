import numpy as np
import cv2
from tensorflow.keras.applications import ResNet101
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet import preprocess_input
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt

# -------------------
# 1. Load ResNet101 and pick an intermediate layer
# -------------------
base_model = ResNet101(weights='imagenet', include_top=False)
# layer_name = 'conv2_block3_out'  # still has good spatial detail
intermediate_model = Model(inputs=base_model.input,outputs=base_model.layers[1].output)

# -------------------
# 2. Load image and preprocess
# -------------------
img_path = "../training_set/training_set/cats/cat.41.jpg"
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
x = np.expand_dims(img_array, axis=0)
x = preprocess_input(x)

# -------------------
# 3. Extract ResNet features
# -------------------
feature_map = intermediate_model.predict(x)[0]  # shape (H, W, C)
channel_idx = 1  # pick any channel index
single_channel = feature_map[:, :, channel_idx]

# -------------------
# 4. Create a pseudo-RGB image from single channel
# -------------------
# normalize channel to [0,255]
norm_channel = cv2.normalize(single_channel, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
rgb_img = cv2.merge([norm_channel]*3)

# -------------------
# 5. Adaptive Median Filter (here simple median filter for simplicity)
# -------------------
smoothed = cv2.medianBlur(rgb_img, 3)

# -------------------
# 6. Weighted sum (2R + 3G + 4B)
# -------------------
R, G, B = cv2.split(smoothed)
weighted_img = 2*R + 3*G + 4*B
weighted_img = weighted_img.astype(np.float32)

# -------------------
# 7. Directional masks (from the paper)
# -------------------
A = np.array([[0,0,0],
              [-1,0,1],
              [0,0,0]], dtype=np.float32)  # 0°
B = np.array([[0,-1,0],
              [0,0,0],
              [0,1,0]], dtype=np.float32)  # 90°
C = np.array([[-1,0,0],
              [0,0,0],
              [0,0,1]], dtype=np.float32)  # 135°
D = np.array([[0,0,-1],
              [0,0,0],
              [1,0,0]], dtype=np.float32)  # 45°

# Convolve in each direction
diff0   = cv2.filter2D(weighted_img, -1, A)
diff90  = cv2.filter2D(weighted_img, -1, B)
diff135 = cv2.filter2D(weighted_img, -1, C)
diff45  = cv2.filter2D(weighted_img, -1, D)

# absolute differences
diff0 = np.abs(diff0)
diff90 = np.abs(diff90)
diff135 = np.abs(diff135)
diff45 = np.abs(diff45)

# -------------------
# 8. Take maximum directional difference
# -------------------
max_diff = np.maximum.reduce([diff0, diff90, diff135, diff45])

# -------------------
# 9. Threshold T = 1.2 * average(max_diff)
# -------------------
T = 1.2 * np.mean(max_diff)
_, edge_map = cv2.threshold(max_diff, T, 255, cv2.THRESH_BINARY)

# -------------------
# 10. Morphological thinning (skeletonization if ximgproc not available)
# -------------------
def morphological_thinning(img):
    img = img.copy().astype(np.uint8)
    skel = np.zeros(img.shape, np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
    while True:
        eroded = cv2.erode(img, kernel)
        temp = cv2.dilate(eroded, kernel)
        temp = cv2.subtract(img, temp)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded.copy()
        if cv2.countNonZero(img) == 0:
            break
    return skel

thinned = morphological_thinning(edge_map)

# -------------------
# 11. Show results
# -------------------
plt.subplot(1,3,1)
plt.imshow(img_array.astype(np.uint8))
plt.title("Original")

plt.subplot(1,3,2)
plt.imshow(norm_channel, cmap='gray')
plt.title("One ResNet Channel")

plt.subplot(1,3,3)
plt.imshow(thinned, cmap='gray')
plt.title("Final Edge Map (Paper-style)")

plt.show()
