import cv2
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
import imutils
from tkinter import *
from tkinter import filedialog

clusters = 5
img = None
org_img = None

root = Tk()
root.title('Image and Cluster Count Input')

image_path = StringVar()


def get_image():
    path = filedialog.askopenfilename(
        initialdir='/',
        title='Select Image File',
        filetypes=(('JPEG Files', '*.jpg'), ('PNG Files', '*.png'), ('All Files', '*.*'))
    )
    return path


def select_image():
    path = get_image()
    if path:
        image_path.set(path)
        path_label.config(text=path.split('/')[-1])


def get_cluster_count():
    global clusters, img, org_img
    try:
        clusters = int(cluster_entry.get())
    except ValueError:
        clusters = 5
    path = image_path.get()
    if not path:
        return
    org_img = cv2.imread(path)
    img = org_img.copy()
    root.destroy()


Label(root, text="Step 1: Select an image").pack(pady=(10, 0))
image_button = Button(root, text='Select Image', command=select_image)
image_button.pack(pady=4)
path_label = Label(root, text="No file selected", fg="gray")
path_label.pack()

Label(root, text="Step 2: Number of dominant colors").pack(pady=(10, 0))
cluster_entry = Entry(root)
cluster_entry.insert(0, "5")
cluster_entry.pack(pady=4)

submit_button = Button(root, text='Submit', command=get_cluster_count)
submit_button.pack(pady=(4, 10))

root.mainloop()

if img is None:
    print("No image selected. Exiting.")
    exit()

# --- Resize and flatten for KMeans ---
img = imutils.resize(img, height=200)
print('After resizing shape --> ', img.shape)

flat_img = np.reshape(img, (-1, 3))
print('After Flattening shape --> ', flat_img.shape)

kmeans = KMeans(n_clusters=clusters, random_state=0)
kmeans.fit(flat_img)

dominant_colors = np.array(kmeans.cluster_centers_, dtype='uint')

percentages = np.unique(kmeans.labels_, return_counts=True)[1] / flat_img.shape[0]
p_and_c = list(zip(percentages, dominant_colors))
p_and_c = sorted(p_and_c, reverse=True)

# --- Plot individual color swatches ---
block = np.ones((50, 50, 3), dtype='uint')
plt.figure(figsize=(12, 8))
for i in range(clusters):
    plt.subplot(1, clusters, i + 1)
    block[:] = p_and_c[i][1][::-1]  # BGR -> RGB for matplotlib
    plt.imshow(block)
    plt.xticks([])
    plt.yticks([])
    plt.xlabel(str(round(p_and_c[i][0] * 100, 2)) + '%')

# --- Plot proportional color bar ---
bar = np.ones((50, 500, 3), dtype='uint')
plt.figure(figsize=(12, 8))
plt.title('Proportions of colors in the image')
start = 0
i = 1
for p, c in p_and_c:
    end = start + int(p * bar.shape[1])
    if i == clusters:
        bar[:, start:] = c[::-1]
    else:
        bar[:, start:end] = c[::-1]
    start = end
    i += 1

plt.imshow(bar)
plt.xticks([])
plt.yticks([])

# --- Overlay dominant colors on the original image ---
rows = 1000  # target width
cols = int(org_img.shape[0] / org_img.shape[1] * rows)  # computed height
img = cv2.resize(org_img, dsize=(rows, cols), interpolation=cv2.INTER_LINEAR)

copy = img.copy()
# Draw a semi-transparent white panel in the center
cx, cy = rows // 2, cols // 2
cv2.rectangle(copy, (cx - 250, cy - 90), (cx + 250, cy + 110), (255, 255, 255), -1)

final = cv2.addWeighted(img, 0.1, copy, 0.9, 0)
cv2.putText(final, 'Most Dominant Colors in the Image',
            (cx - 230, cy - 40),
            cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 0), 1, cv2.LINE_AA)

# Draw color swatches vertically in the panel
start = cy - 220
for i in range(min(5, clusters)):
    end = start + 70
    color = p_and_c[i][1]  # BGR
    final[start:end, cx:cx + 70] = color
    cv2.putText(final, str(i + 1),
                (start + 25, cx + 45),
                cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)
    start = end + 20

plt.show()
cv2.imshow('Image', img)
cv2.imshow('img', final)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('output.png', final)
