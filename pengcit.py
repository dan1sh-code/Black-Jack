import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd

# === Ganti path folder sesuai lokasi gambar kamu ===
folder_path = r"D:\pengolahan citra\Pertemuan 1"  # contoh path, sesuaikan sendiri

# === Threshold untuk binary ===
threshold = 100  

# 1. Cek folder
if not os.path.exists(folder_path):
    print("❌ Folder tidak ditemukan:", folder_path)
else:
    print("📂 Folder ditemukan:", folder_path)

    # 2. Ambil semua file gambar
    img_files = [f for f in os.listdir(folder_path) if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    if not img_files:
        print("❌ Tidak ada file gambar ditemukan di folder ini.")
    else:
        print("✅ File gambar ditemukan:", img_files)

        # 3. Proses setiap file
        for idx, fname in enumerate(img_files, start=1):
            path = os.path.join(folder_path, fname)

            try:
                # === True Color (RGB) ===
                img = Image.open(path).convert("RGB")
                rgb_np = np.array(img, dtype=np.uint8)

                # === Grayscale ===
                gray_img = img.convert("L")
                gray_np = np.array(gray_img, dtype=np.uint8)

                # === Binary (threshold) ===
                binary_np = (gray_np >= threshold).astype(np.uint8)

                # === Tampilkan citra ===
                fig, axs = plt.subplots(1, 3, figsize=(15, 5))
                fig.suptitle(f"Gambar {idx}: {fname}", fontsize=14)

                axs[0].imshow(rgb_np)
                axs[0].axis('off')
                axs[0].set_title("True Color (RGB)")

                axs[1].imshow(gray_np, cmap='gray')
                axs[1].axis('off')
                axs[1].set_title("Grayscale (8-bit)")

                axs[2].imshow(binary_np, cmap='gray')
                axs[2].axis('off')
                axs[2].set_title("Binary (0/1)")

                plt.tight_layout()
                plt.show(block=False)
                plt.pause(3)  # tampilkan 3 detik per gambar

                # === Cuplikan piksel (5x3) ===
                h, w = gray_np.shape
                sample_h = min(5, h)
                sample_w = min(3, w)

                print(f"\n=== Cuplikan Nilai Piksel Gambar {idx} ({fname}) ===")
                print("R channel:\n", rgb_np[:sample_h, :sample_w, 0])
                print("G channel:\n", rgb_np[:sample_h, :sample_w, 1])
                print("B channel:\n", rgb_np[:sample_h, :sample_w, 2])
                print("Grayscale:\n", gray_np[:sample_h, :sample_w])
                print("Binary (0/1):\n", binary_np[:sample_h, :sample_w])

                # === Ringkasan ===
                summary = {
                    "Nama File": fname,
                    "Ukuran (H x W)": f"{h} x {w} piksel",
                    "Rentang Grayscale": f"{int(gray_np.min())} .. {int(gray_np.max())}",
                    "Proporsi Putih Binary": f"{binary_np.mean():.3f}",
                    "Threshold Biner": threshold
                }
                summary_df = pd.DataFrame([summary])
                print(f"\n=== Ringkasan Gambar {idx} ===")
                print(summary_df)

            except Exception as e:
                print(f"⚠️ Gagal memproses {fname}: {e}")

input("\nTekan Enter untuk menutup program...")
