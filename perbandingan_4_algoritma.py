# ============================================================
# VISUALISASI INTERAKTIF 4 ALGORITMA SUPERVISED MACHINE LEARNING
# TANPA SAVE GAMBAR - VISUAL LANGSUNG KELUAR DI LAYAR
# ============================================================

import time
import warnings
from textwrap import fill

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

warnings.filterwarnings("ignore")

# ============================================================
# KONFIGURASI
# ============================================================

RANDOM_STATE = 42
TEST_SIZE = 0.3

# Supaya objek animasi tidak hilang dari memory
ANIMATIONS = []

try:
    plt.style.use("seaborn-v0_8-whitegrid")
except:
    plt.style.use("default")

sns.set_palette("Set2")


# ============================================================
# FUNGSI BANTUAN
# ============================================================

def create_models():
    """
    Membuat 4 algoritma supervised machine learning classification.
    Logistic Regression dan SVM memakai StandardScaler karena sensitif terhadap skala fitur.
    """

    models = {
        "Logistic Regression": {
            "model": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(
                    max_iter=3000,
                    random_state=RANDOM_STATE
                ))
            ]),
            "color": "#FF6B6B",
            "short": "Model linear berbasis probabilitas",
            "desc": "Menghitung probabilitas kelas menggunakan fungsi sigmoid."
        },

        "Decision Tree": {
            "model": DecisionTreeClassifier(
                max_depth=5,
                random_state=RANDOM_STATE
            ),
            "color": "#4ECDC4",
            "short": "Model berbasis pohon keputusan",
            "desc": "Membagi data dengan aturan IF-ELSE sampai menghasilkan keputusan."
        },

        "Random Forest": {
            "model": RandomForestClassifier(
                n_estimators=200,
                random_state=RANDOM_STATE,
                n_jobs=-1
            ),
            "color": "#45B7D1",
            "short": "Gabungan banyak Decision Tree",
            "desc": "Menggabungkan banyak pohon keputusan lalu mengambil voting mayoritas."
        },

        "SVM": {
            "model": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", SVC(
                    kernel="rbf",
                    C=1.0,
                    gamma="scale",
                    random_state=RANDOM_STATE
                ))
            ]),
            "color": "#9B5DE5",
            "short": "Model margin maksimum",
            "desc": "Mencari batas pemisah terbaik dengan margin sebesar mungkin."
        }
    }

    return models


def show_figure(fig, title=""):
    """
    Menampilkan figure tanpa menyimpan gambar.
    Tutup window gambar untuk lanjut ke visual berikutnya.
    """

    if title:
        print(f"\nMenampilkan: {title}")
        print("Tutup window gambar untuk lanjut ke visual berikutnya...")

    try:
        fig.canvas.manager.set_window_title(title)
    except:
        pass

    plt.show(block=True)


def empty_offsets():
    """
    Untuk scatter kosong pada animasi.
    """
    return np.empty((0, 2))


def print_algorithm_explanation():
    """
    Menampilkan penjelasan algoritma di terminal.
    """

    print("\n" + "=" * 80)
    print("ALGORITMA YANG DIPAKAI")
    print("=" * 80)

    print("""
1. Logistic Regression
   - Jenis      : Supervised Learning - Classification
   - Cara kerja : Menghitung probabilitas kelas menggunakan fungsi sigmoid.
   - Cocok      : Data yang pola pemisahnya cukup linear.
   - Kelebihan  : Cepat, sederhana, dan mudah dijelaskan.
   - Kekurangan : Kurang kuat untuk pola data yang sangat non-linear.

2. Decision Tree
   - Jenis      : Supervised Learning - Classification
   - Cara kerja : Membagi data dengan aturan bercabang seperti IF-ELSE.
   - Contoh     : Jika fitur A > nilai tertentu, masuk cabang kanan, selain itu cabang kiri.
   - Kelebihan  : Sangat mudah dipahami dan divisualisasikan.
   - Kekurangan : Rentan overfitting jika pohonnya terlalu dalam.

3. Random Forest
   - Jenis      : Supervised Learning - Classification, Ensemble Method
   - Cara kerja : Membuat banyak Decision Tree, lalu hasil akhir ditentukan dengan voting mayoritas.
   - Kelebihan  : Lebih stabil dan biasanya lebih akurat daripada satu Decision Tree.
   - Kekurangan : Lebih berat dan lebih sulit dijelaskan secara detail.

4. SVM atau Support Vector Machine
   - Jenis      : Supervised Learning - Classification
   - Cara kerja : Mencari garis/bidang pemisah terbaik dengan margin terbesar.
   - Kernel RBF : Membantu SVM menangani pola data yang tidak linear.
   - Kelebihan  : Kuat untuk data kompleks.
   - Kekurangan : Sensitif terhadap scaling dan parameter.
""")


# ============================================================
# 1. LOAD DATASET
# ============================================================

print("=" * 80)
print("VISUALISASI INTERAKTIF 4 ALGORITMA SUPERVISED MACHINE LEARNING")
print("TANPA SAVE GAMBAR - GAMBAR DAN ANIMASI LANGSUNG KELUAR")
print("=" * 80)

print_algorithm_explanation()

data = load_breast_cancer()

X = data.data
y = data.target

feature_names = data.feature_names
target_names = data.target_names

print("\n" + "=" * 80)
print("DATASET YANG DIGUNAKAN")
print("=" * 80)

print("Dataset       : Breast Cancer Wisconsin")
print(f"Jumlah data   : {X.shape[0]} baris")
print(f"Jumlah fitur  : {X.shape[1]} fitur")
print(f"Jumlah kelas  : {len(np.unique(y))}")
print(f"Nama kelas    : {target_names}")

print("""
Keterangan kelas:
0 = malignant / kanker ganas
1 = benign    / kanker jinak
""")

class_counts = pd.Series(y).value_counts().sort_index()

print("Distribusi kelas:")
print(class_counts)


# ============================================================
# 2. SPLIT DATA
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

print("\n" + "=" * 80)
print("SPLIT DATA")
print("=" * 80)
print(f"Training data : {X_train.shape}")
print(f"Testing data  : {X_test.shape}")


# ============================================================
# 3. PCA UNTUK VISUALISASI 2D
# ============================================================
# Dataset asli punya 30 fitur.
# Supaya semua data bisa divisualisasikan dalam bidang 2D,
# digunakan PCA untuk merangkum 30 fitur menjadi 2 komponen utama.

scaler_pca = StandardScaler()
X_train_scaled_for_pca = scaler_pca.fit_transform(X_train)
X_test_scaled_for_pca = scaler_pca.transform(X_test)
X_all_scaled_for_pca = scaler_pca.transform(X)

pca = PCA(n_components=2, random_state=RANDOM_STATE)

X_train_pca = pca.fit_transform(X_train_scaled_for_pca)
X_test_pca = pca.transform(X_test_scaled_for_pca)
X_all_pca = pca.transform(X_all_scaled_for_pca)

explained_var = pca.explained_variance_ratio_.sum()

print("\n" + "=" * 80)
print("PCA UNTUK VISUALISASI")
print("=" * 80)
print("Dataset punya 30 fitur, jadi PCA dipakai untuk merangkum data ke 2 dimensi.")
print(f"Informasi utama yang tertangkap oleh PCA 2D: {explained_var:.2%}")


# ============================================================
# 4. TRAINING DAN EVALUASI MODEL PADA DATA ASLI 30 FITUR
# ============================================================

models = create_models()

results = {}
predictions = {}

print("\n" + "=" * 80)
print("TRAINING DAN EVALUASI 4 MODEL")
print("=" * 80)

for name, info in models.items():
    print(f"\nTraining {name}...")

    model = info["model"]

    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time

    y_pred = model.predict(X_test)
    predictions[name] = y_pred

    results[name] = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1-Score": f1_score(y_test, y_pred, zero_division=0),
        "Training Time": training_time
    }

    print(f"Accuracy      : {results[name]['Accuracy']:.4f}")
    print(f"Precision     : {results[name]['Precision']:.4f}")
    print(f"Recall        : {results[name]['Recall']:.4f}")
    print(f"F1-Score      : {results[name]['F1-Score']:.4f}")
    print(f"Training Time : {training_time:.5f} detik")

results_df = pd.DataFrame(results).T
model_names = results_df.index.tolist()

print("\nRingkasan hasil:")
print(results_df.round(4))


# ============================================================
# 5. TRAINING MODEL DENGAN DATA PCA 2D UNTUK DECISION BOUNDARY
# ============================================================

models_2d = create_models()

for name, info in models_2d.items():
    info["model"].fit(X_train_pca, y_train)


# ============================================================
# 6. VISUAL 1 - DASHBOARD LENGKAP YANG RAPI
# ============================================================

fig = plt.figure(figsize=(20, 11), constrained_layout=True)
fig.patch.set_facecolor("#F8FAFC")

gs = fig.add_gridspec(
    3,
    4,
    height_ratios=[0.55, 1.45, 1.15],
    width_ratios=[1, 1, 1, 1]
)

# ------------------------------------------------------------
# HEADER
# ------------------------------------------------------------

ax_header = fig.add_subplot(gs[0, :])
ax_header.axis("off")

ax_header.text(
    0.01,
    0.78,
    "Dashboard Perbandingan 4 Algoritma Classification",
    fontsize=25,
    fontweight="bold",
    color="#0F172A",
    ha="left",
    va="center",
    transform=ax_header.transAxes
)

ax_header.text(
    0.01,
    0.38,
    "Dataset: Breast Cancer Wisconsin  |  Visualisasi 2D memakai PCA  |  Model: Logistic Regression, Decision Tree, Random Forest, SVM",
    fontsize=13,
    color="#475569",
    ha="left",
    va="center",
    transform=ax_header.transAxes
)

info_cards = [
    ("Jumlah Data", f"{X.shape[0]}", "#E0F2FE"),
    ("Jumlah Fitur", f"{X.shape[1]}", "#DCFCE7"),
    ("Malignant", f"{class_counts.loc[0]}", "#FEE2E2"),
    ("Benign", f"{class_counts.loc[1]}", "#DBEAFE"),
    ("Info PCA 2D", f"{explained_var:.2%}", "#F3E8FF")
]

start_x = 0.50
gap = 0.095

for i, (label, value, color) in enumerate(info_cards):
    x = start_x + i * gap

    ax_header.text(
        x,
        0.70,
        value,
        fontsize=16,
        fontweight="bold",
        color="#0F172A",
        ha="center",
        va="center",
        transform=ax_header.transAxes,
        bbox=dict(
            boxstyle="round,pad=0.45",
            fc=color,
            ec="#CBD5E1",
            lw=1.5
        )
    )

    ax_header.text(
        x,
        0.27,
        label,
        fontsize=10,
        color="#475569",
        ha="center",
        va="center",
        transform=ax_header.transAxes
    )

# ------------------------------------------------------------
# PANEL 1: Scatter PCA
# ------------------------------------------------------------

ax_scatter = fig.add_subplot(gs[1, 0:2])
ax_scatter.set_facecolor("white")

ax_scatter.scatter(
    X_all_pca[y == 0, 0],
    X_all_pca[y == 0, 1],
    color="#FF6B6B",
    edgecolor="black",
    linewidth=0.4,
    s=42,
    alpha=0.78,
    label="Malignant / Ganas"
)

ax_scatter.scatter(
    X_all_pca[y == 1, 0],
    X_all_pca[y == 1, 1],
    color="#45B7D1",
    edgecolor="black",
    linewidth=0.4,
    s=42,
    alpha=0.78,
    label="Benign / Jinak"
)

ax_scatter.set_title(
    "Sebaran Semua Data dalam PCA 2D",
    fontsize=15,
    fontweight="bold",
    pad=12
)

ax_scatter.set_xlabel("PCA Component 1")
ax_scatter.set_ylabel("PCA Component 2")
ax_scatter.legend(frameon=True)
ax_scatter.grid(alpha=0.25)

# ------------------------------------------------------------
# PANEL 2: Heatmap Performa
# ------------------------------------------------------------

ax_heat = fig.add_subplot(gs[1, 2:4])
ax_heat.set_facecolor("white")

sns.heatmap(
    results_df[["Accuracy", "Precision", "Recall", "F1-Score"]],
    annot=True,
    fmt=".4f",
    cmap="YlGnBu",
    linewidths=1,
    linecolor="white",
    cbar_kws={"shrink": 0.8},
    ax=ax_heat
)

ax_heat.set_title(
    "Heatmap Performa Model",
    fontsize=15,
    fontweight="bold",
    pad=12
)

ax_heat.set_xlabel("Metrik Evaluasi")
ax_heat.set_ylabel("Algoritma")

# ------------------------------------------------------------
# PANEL 3: Accuracy Bar
# ------------------------------------------------------------

ax_acc = fig.add_subplot(gs[2, 0])
ax_acc.set_facecolor("white")

colors = [models[name]["color"] for name in model_names]

bars = ax_acc.bar(
    model_names,
    results_df["Accuracy"],
    color=colors,
    edgecolor="black",
    linewidth=1
)

ax_acc.set_title("Accuracy", fontsize=14, fontweight="bold", pad=10)
ax_acc.set_ylim(0, 1.08)
ax_acc.set_ylabel("Score")
ax_acc.tick_params(axis="x", rotation=25)
ax_acc.grid(axis="y", alpha=0.25)

for bar in bars:
    h = bar.get_height()
    ax_acc.text(
        bar.get_x() + bar.get_width() / 2,
        h + 0.015,
        f"{h:.4f}",
        ha="center",
        fontsize=9,
        fontweight="bold"
    )

# ------------------------------------------------------------
# PANEL 4: Training Time
# ------------------------------------------------------------

ax_time = fig.add_subplot(gs[2, 1])
ax_time.set_facecolor("white")

bars_time = ax_time.bar(
    model_names,
    results_df["Training Time"],
    color=colors,
    edgecolor="black",
    linewidth=1
)

ax_time.set_title("Training Time", fontsize=14, fontweight="bold", pad=10)
ax_time.set_ylabel("Detik")
ax_time.tick_params(axis="x", rotation=25)
ax_time.grid(axis="y", alpha=0.25)

max_time = results_df["Training Time"].max()
offset = max_time * 0.06 if max_time > 0 else 0.001

for bar in bars_time:
    h = bar.get_height()
    ax_time.text(
        bar.get_x() + bar.get_width() / 2,
        h + offset,
        f"{h:.5f}s",
        ha="center",
        fontsize=8,
        fontweight="bold"
    )

# ------------------------------------------------------------
# PANEL 5: Confusion Matrix Model Terbaik
# ------------------------------------------------------------

best_model_name = results_df["Accuracy"].idxmax()
best_cm = confusion_matrix(y_test, predictions[best_model_name])

ax_cm = fig.add_subplot(gs[2, 2])
ax_cm.set_facecolor("white")

sns.heatmap(
    best_cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    linewidths=1,
    linecolor="white",
    cbar=False,
    xticklabels=["Malignant", "Benign"],
    yticklabels=["Malignant", "Benign"],
    ax=ax_cm
)

ax_cm.set_title(
    f"Confusion Matrix\nModel Terbaik: {best_model_name}",
    fontsize=13,
    fontweight="bold",
    pad=10
)

ax_cm.set_xlabel("Predicted")
ax_cm.set_ylabel("Actual")

# ------------------------------------------------------------
# PANEL 6: Ringkasan Kesimpulan
# ------------------------------------------------------------

ax_summary = fig.add_subplot(gs[2, 3])
ax_summary.set_facecolor("white")
ax_summary.axis("off")

best_accuracy = results_df["Accuracy"].idxmax()
best_f1 = results_df["F1-Score"].idxmax()
fastest_model = results_df["Training Time"].idxmin()

summary_text = f"""
RINGKASAN HASIL

Akurasi Tertinggi:
{best_accuracy}
Score: {results_df.loc[best_accuracy, 'Accuracy']:.4f}

F1-Score Tertinggi:
{best_f1}
Score: {results_df.loc[best_f1, 'F1-Score']:.4f}

Training Tercepat:
{fastest_model}
Waktu: {results_df.loc[fastest_model, 'Training Time']:.5f}s

Catatan:
- PCA hanya untuk visualisasi 2D.
- Evaluasi model tetap memakai 30 fitur asli.
"""

ax_summary.text(
    0.05,
    0.95,
    summary_text,
    fontsize=12,
    color="#0F172A",
    va="top",
    ha="left",
    linespacing=1.35,
    bbox=dict(
        boxstyle="round,pad=0.7",
        fc="white",
        ec="#CBD5E1",
        lw=1.8
    ),
    transform=ax_summary.transAxes
)

fig.suptitle(
    "Visualisasi Lengkap Dataset dan Performa Model",
    fontsize=22,
    fontweight="bold",
    color="#0F172A",
    y=1.02
)

show_figure(fig, "Dashboard Lengkap yang Rapi")


# ============================================================
# 7. VISUAL 2 - ANIMASI SEMUA DATA MUNCUL SATU PER SATU
# ============================================================

fig, ax = plt.subplots(figsize=(12, 8))
fig.patch.set_facecolor("#F8FAFC")

ax.set_title(
    "Animasi Semua Data Breast Cancer\n30 Fitur Dirangkum ke PCA 2D",
    fontsize=18,
    fontweight="bold"
)

ax.set_xlabel("PCA Component 1")
ax.set_ylabel("PCA Component 2")

x_margin = 1.0
y_margin = 1.0

ax.set_xlim(X_all_pca[:, 0].min() - x_margin, X_all_pca[:, 0].max() + x_margin)
ax.set_ylim(X_all_pca[:, 1].min() - y_margin, X_all_pca[:, 1].max() + y_margin)
ax.grid(alpha=0.3)

scatter_malignant = ax.scatter(
    [],
    [],
    c="#FF6B6B",
    edgecolor="black",
    s=50,
    alpha=0.8,
    label="Malignant"
)

scatter_benign = ax.scatter(
    [],
    [],
    c="#45B7D1",
    edgecolor="black",
    s=50,
    alpha=0.8,
    label="Benign"
)

ax.legend(loc="upper right")

info_box = ax.text(
    0.02,
    0.96,
    "",
    transform=ax.transAxes,
    fontsize=13,
    fontweight="bold",
    va="top",
    bbox=dict(boxstyle="round", fc="white", ec="#CBD5E1")
)

rng = np.random.default_rng(RANDOM_STATE)
order = rng.permutation(len(X_all_pca))

n_frames = 100
n_data = len(X_all_pca)
step = int(np.ceil(n_data / n_frames))

def update_data_animation(frame):
    k = min((frame + 1) * step, n_data)
    selected_idx = order[:k]

    idx_malignant = selected_idx[y[selected_idx] == 0]
    idx_benign = selected_idx[y[selected_idx] == 1]

    scatter_malignant.set_offsets(
        X_all_pca[idx_malignant] if len(idx_malignant) > 0 else empty_offsets()
    )

    scatter_benign.set_offsets(
        X_all_pca[idx_benign] if len(idx_benign) > 0 else empty_offsets()
    )

    info_box.set_text(
        f"Data tampil: {k}/{n_data}\n"
        f"Malignant: {len(idx_malignant)}\n"
        f"Benign: {len(idx_benign)}\n"
        f"PCA info: {explained_var:.2%}"
    )

    return scatter_malignant, scatter_benign, info_box

anim1 = FuncAnimation(
    fig,
    update_data_animation,
    frames=n_frames,
    interval=60,
    blit=False,
    repeat=True
)

ANIMATIONS.append(anim1)

print("\nMenampilkan animasi semua data...")
print("Tutup window animasi untuk lanjut.")
plt.show(block=True)


# ============================================================
# 8. VISUAL 3 - ANIMASI DECISION BOUNDARY 4 MODEL
# ============================================================

x_min, x_max = X_all_pca[:, 0].min() - 1.5, X_all_pca[:, 0].max() + 1.5
y_min, y_max = X_all_pca[:, 1].min() - 1.5, X_all_pca[:, 1].max() + 1.5

xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 300),
    np.linspace(y_min, y_max, 300)
)

grid = np.c_[xx.ravel(), yy.ravel()]

boundary_images = {}

for name, info in models_2d.items():
    Z = info["model"].predict(grid)
    Z = Z.reshape(xx.shape)

    rgb = np.zeros((Z.shape[0], Z.shape[1], 3))
    rgb[Z == 0] = np.array([255, 107, 107]) / 255.0
    rgb[Z == 1] = np.array([69, 183, 209]) / 255.0

    boundary_images[name] = rgb

fig, axes = plt.subplots(2, 2, figsize=(16, 11))
fig.patch.set_facecolor("#F8FAFC")
axes = axes.ravel()

images = {}
scatters_0 = {}
scatters_1 = {}
texts = {}

for idx, name in enumerate(model_names):
    ax = axes[idx]

    img = ax.imshow(
        boundary_images[name],
        extent=[x_min, x_max, y_min, y_max],
        origin="lower",
        alpha=0.0,
        aspect="auto"
    )

    sc0 = ax.scatter(
        [],
        [],
        c="#FF6B6B",
        edgecolor="black",
        s=35,
        alpha=0.85,
        label="Malignant"
    )

    sc1 = ax.scatter(
        [],
        [],
        c="#45B7D1",
        edgecolor="black",
        s=35,
        alpha=0.85,
        label="Benign"
    )

    txt = ax.text(
        0.03,
        0.95,
        "",
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        va="top",
        bbox=dict(boxstyle="round", fc="white", ec="#CBD5E1")
    )

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_title(name, fontsize=15, fontweight="bold", color=models[name]["color"])
    ax.set_xlabel("PCA Component 1")
    ax.set_ylabel("PCA Component 2")
    ax.grid(alpha=0.25)

    images[name] = img
    scatters_0[name] = sc0
    scatters_1[name] = sc1
    texts[name] = txt

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper right")

fig.suptitle(
    "Animasi Decision Boundary 4 Algoritma\nArea Warna = Prediksi Model, Titik = Data Asli",
    fontsize=21,
    fontweight="bold"
)

n_frames = 110
step_boundary = int(np.ceil(n_data / 60))

def update_boundary_animation(frame):
    if frame < 60:
        k = min((frame + 1) * step_boundary, n_data)
        alpha_boundary = 0.0
    else:
        k = n_data
        alpha_boundary = min((frame - 60) / 30, 0.35)

    selected_idx = order[:k]

    idx_malignant = selected_idx[y[selected_idx] == 0]
    idx_benign = selected_idx[y[selected_idx] == 1]

    artists = []

    for name in model_names:
        scatters_0[name].set_offsets(
            X_all_pca[idx_malignant] if len(idx_malignant) > 0 else empty_offsets()
        )

        scatters_1[name].set_offsets(
            X_all_pca[idx_benign] if len(idx_benign) > 0 else empty_offsets()
        )

        images[name].set_alpha(alpha_boundary)

        texts[name].set_text(
            f"Data: {k}/{n_data}\n"
            f"Accuracy: {results_df.loc[name, 'Accuracy']:.4f}\n"
            f"F1: {results_df.loc[name, 'F1-Score']:.4f}"
        )

        artists.extend([
            scatters_0[name],
            scatters_1[name],
            images[name],
            texts[name]
        ])

    return artists

anim2 = FuncAnimation(
    fig,
    update_boundary_animation,
    frames=n_frames,
    interval=60,
    blit=False,
    repeat=True
)

ANIMATIONS.append(anim2)

print("\nMenampilkan animasi decision boundary...")
print("Tutup window animasi untuk lanjut.")
plt.show(block=True)


# ============================================================
# 9. VISUAL 4 - ANIMASI BAR METRIK EVALUASI
# ============================================================

metrics_to_show = ["Accuracy", "Precision", "Recall", "F1-Score"]

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.patch.set_facecolor("#F8FAFC")
axes = axes.ravel()

bar_containers = {}
value_texts = {}

for idx, metric in enumerate(metrics_to_show):
    ax = axes[idx]

    bars = ax.bar(
        model_names,
        [0] * len(model_names),
        color=colors,
        edgecolor="black"
    )

    ax.set_title(metric, fontsize=16, fontweight="bold")
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score")
    ax.tick_params(axis="x", rotation=18)
    ax.grid(axis="y", alpha=0.3)

    texts = []

    for bar in bars:
        txt = ax.text(
            bar.get_x() + bar.get_width() / 2,
            0.02,
            "",
            ha="center",
            fontsize=11,
            fontweight="bold"
        )
        texts.append(txt)

    bar_containers[metric] = bars
    value_texts[metric] = texts

fig.suptitle(
    "Animasi Perbandingan Metrik Evaluasi 4 Model",
    fontsize=21,
    fontweight="bold"
)

n_frames_metric = 80

def ease_out_cubic(t):
    return 1 - (1 - t) ** 3

def update_metric_animation(frame):
    t = frame / (n_frames_metric - 1)
    progress = ease_out_cubic(t)

    artists = []

    for metric in metrics_to_show:
        target_values = results_df[metric].values

        for i, bar in enumerate(bar_containers[metric]):
            value = target_values[i] * progress
            bar.set_height(value)

            value_texts[metric][i].set_text(f"{value:.3f}")
            value_texts[metric][i].set_y(value + 0.02)

            artists.append(bar)
            artists.append(value_texts[metric][i])

    return artists

anim3 = FuncAnimation(
    fig,
    update_metric_animation,
    frames=n_frames_metric,
    interval=55,
    blit=False,
    repeat=True
)

ANIMATIONS.append(anim3)

print("\nMenampilkan animasi metrik evaluasi...")
print("Tutup window animasi untuk lanjut.")
plt.show(block=True)


# ============================================================
# 10. VISUAL 5 - PERBEDAAN 4 ALGORITMA DALAM BENTUK CARD RAPI
# ============================================================

fig, axes = plt.subplots(2, 2, figsize=(18, 10), constrained_layout=True)
fig.patch.set_facecolor("#F8FAFC")

axes = axes.ravel()

algorithm_cards = [
    {
        "name": "Logistic Regression",
        "color": models["Logistic Regression"]["color"],
        "icon": "1",
        "cara": "Menghitung probabilitas sebuah data masuk ke kelas tertentu menggunakan fungsi sigmoid.",
        "plus": "Cepat, sederhana, ringan, dan mudah dijelaskan.",
        "minus": "Kurang kuat jika pola data sangat kompleks atau sangat non-linear.",
        "cocok": "Baseline model, data yang relatif linear, dan kebutuhan interpretasi cepat."
    },
    {
        "name": "Decision Tree",
        "color": models["Decision Tree"]["color"],
        "icon": "2",
        "cara": "Membagi data menjadi aturan bercabang seperti IF-ELSE sampai menghasilkan keputusan akhir.",
        "plus": "Sangat mudah dipahami dan divisualisasikan seperti pohon keputusan.",
        "minus": "Rentan overfitting jika kedalaman pohon tidak dibatasi.",
        "cocok": "Analisis berbasis aturan, penjelasan keputusan, dan pembelajaran konsep klasifikasi."
    },
    {
        "name": "Random Forest",
        "color": models["Random Forest"]["color"],
        "icon": "3",
        "cara": "Membuat banyak Decision Tree lalu menggabungkan hasilnya dengan voting mayoritas.",
        "plus": "Lebih stabil, biasanya lebih akurat, dan mampu mengurangi overfitting.",
        "minus": "Lebih berat dibanding satu Decision Tree dan kurang mudah dijelaskan secara detail.",
        "cocok": "Kasus yang mengejar akurasi tinggi dan dataset dengan banyak fitur."
    },
    {
        "name": "SVM",
        "color": models["SVM"]["color"],
        "icon": "4",
        "cara": "Mencari hyperplane atau batas pemisah terbaik dengan margin terbesar antar kelas.",
        "plus": "Kuat untuk data kompleks, apalagi dengan kernel seperti RBF.",
        "minus": "Sensitif terhadap scaling dan pemilihan parameter seperti C dan gamma.",
        "cocok": "Data berdimensi tinggi, pola non-linear, dan boundary yang kompleks."
    }
]

for ax, item in zip(axes, algorithm_cards):
    ax.set_facecolor("white")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#CBD5E1")
        spine.set_linewidth(2)

    # Header warna
    ax.add_patch(
        Rectangle(
            (0, 0.82),
            1,
            0.18,
            color=item["color"],
            transform=ax.transAxes,
            clip_on=False
        )
    )

    # Nomor/icon
    ax.text(
        0.05,
        0.91,
        item["icon"],
        fontsize=18,
        fontweight="bold",
        color="white",
        ha="center",
        va="center",
        bbox=dict(
            boxstyle="circle,pad=0.35",
            fc="#0F172A",
            ec="white",
            lw=1.5
        ),
        transform=ax.transAxes
    )

    # Nama algoritma
    ax.text(
        0.13,
        0.91,
        item["name"],
        fontsize=17,
        fontweight="bold",
        color="#0F172A",
        ha="left",
        va="center",
        transform=ax.transAxes
    )

    y_pos = 0.73

    sections = [
        ("Cara kerja", item["cara"]),
        ("Kelebihan", item["plus"]),
        ("Kekurangan", item["minus"]),
        ("Cocok untuk", item["cocok"])
    ]

    for title, content in sections:
        ax.text(
            0.06,
            y_pos,
            title,
            fontsize=12,
            fontweight="bold",
            color="#0F172A",
            ha="left",
            va="top",
            transform=ax.transAxes
        )

        ax.text(
            0.06,
            y_pos - 0.06,
            fill(content, width=58),
            fontsize=11,
            color="#334155",
            ha="left",
            va="top",
            linespacing=1.35,
            transform=ax.transAxes
        )

        y_pos -= 0.20

fig.suptitle(
    "Perbedaan 4 Algoritma Supervised Machine Learning Classification",
    fontsize=23,
    fontweight="bold",
    color="#0F172A",
    y=1.03
)

show_figure(fig, "Perbedaan 4 Algoritma yang Rapi")


# ============================================================
# 11. KESIMPULAN AKHIR DI TERMINAL
# ============================================================

best_accuracy = results_df["Accuracy"].idxmax()
best_f1 = results_df["F1-Score"].idxmax()
fastest_model = results_df["Training Time"].idxmin()

print("\n" + "=" * 80)
print("KESIMPULAN AKHIR")
print("=" * 80)

print(f"Model dengan Accuracy tertinggi : {best_accuracy}")
print(f"Model dengan F1-Score tertinggi : {best_f1}")
print(f"Model training tercepat         : {fastest_model}")

print("\nPerbedaan utama:")
print("""
1. Logistic Regression
   - Paling sederhana dan cepat.
   - Cocok sebagai model awal atau baseline.

2. Decision Tree
   - Paling mudah dipahami karena seperti aturan IF-ELSE.
   - Cocok jika ingin menjelaskan proses keputusan model.

3. Random Forest
   - Lebih stabil karena gabungan banyak pohon.
   - Biasanya performanya bagus untuk banyak kasus.

4. SVM
   - Bagus untuk pola yang kompleks.
   - Perlu scaling dan pemilihan parameter yang tepat.
""")

print("=" * 80)
print("SELESAI")
print("=" * 80)