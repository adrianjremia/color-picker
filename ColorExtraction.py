import cv2
import numpy as np
from sklearn.cluster import KMeans

def extract_dominant_colors(image, k=5):
    # Konversi gambar ke array warna
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = img.reshape((-1, 3))

    # Menggunakan KMeans untuk menemukan warna dominan
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(img)
    
    # Mengambil warna dan frekuensinya
    colors = kmeans.cluster_centers_.astype(int)
    labels = kmeans.labels_
    counts = np.bincount(labels)

    # Mengurutkan warna berdasarkan frekuensi
    sorted_indices = np.argsort(counts)[::-1]
    dominant_colors = colors[sorted_indices]

    return dominant_colors
