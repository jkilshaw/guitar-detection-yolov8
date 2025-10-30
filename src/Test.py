import cv2
import matplotlib.pyplot as plt

# pick one validation image and label
img_path = "data/kaggle/Guitar-Detection-2/valid/images/00816594c8237a7e_jpg.rf.3c8978bd88d42931fc4c1385d481b5a6.jpg"
label_path = img_path.replace("images", "labels").replace(".jpg", ".txt")

# load image
img = cv2.imread(img_path)
h, w, _ = img.shape

# read label
with open(label_path) as f:
    for line in f:
        cls, x, y, bw, bh = map(float, line.split())
        x1, y1 = int((x - bw/2) * w), int((y - bh/2) * h)
        x2, y2 = int((x + bw/2) * w), int((y + bh/2) * h)
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

# show it
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()