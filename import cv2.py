import cv2
import matplotlib.pyplot as plt

# 1. Membaca gambar
img = cv2.imread("gambar.jpg")  # ganti dengan nama file gambar yang ada
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # ubah ke RGB agar sesuai untuk matplotlib

# 2. Mengubah ke grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 3. Memberikan efek blur
blur = cv2.GaussianBlur(gray, (7, 7), 0)

# 4. Deteksi tepi dengan Canny
edges = cv2.Canny(blur, 100, 200)

# 5. Tampilkan hasil
plt.figure(figsize=(10,8))

plt.subplot(2,2,1)
plt.imshow(img_rgb)
plt.title("Citra Asli")
plt.axis("off")

plt.subplot(2,2,2)
plt.imshow(gray, cmap="gray")
plt.title("Grayscale")
plt.axis("off")

plt.subplot(2,2,3)
plt.imshow(blur, cmap="gray")
plt.title("Gaussian Blur")
plt.axis("off")

plt.subplot(2,2,4)
plt.imshow(edges, cmap="gray")
plt.title("Deteksi Tepi (Canny)")
plt.axis("off")

plt.show()
