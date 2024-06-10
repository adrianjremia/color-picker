import streamlit as st
from sklearn.cluster import KMeans
import cv2
import numpy as np
from collections import Counter
import base64

def rgb_to_hex(rgb):
    return '#%02x%02x%02x' % rgb

def get_palette(image, n_colors=5):
    # Reshape image to be a list of pixels
    pixels = image.reshape(-1, 3)

    # Fit the KMeans model
    kmeans = KMeans(n_clusters=n_colors)
    kmeans.fit(pixels)

    # Get the number of pixels for each cluster
    counts = Counter(kmeans.labels_)

    # Get the colors (RGB)
    center_colors = kmeans.cluster_centers_

    # Sort colors by frequency
    sorted_indices = [i[0] for i in sorted(enumerate(counts.values()), key=lambda x: x[1], reverse=True)]
    ordered_colors = [center_colors[i] for i in sorted_indices]
    hex_colors = [rgb_to_hex(tuple(map(int, color))) for color in ordered_colors]

    return ordered_colors, hex_colors

def main():
    st.set_page_config(page_title="Dominant Color Picker", layout="centered", initial_sidebar_state="expanded")
    st.markdown(
        """
        <style>
        body {
            background-color: white;
        }
        .uploadedImage {
            border: 5px solid #ddd;
            border-radius: 10px;
            padding: 10px;
            display: block;
            margin-left: auto;
            margin-right: auto;
            width: 100%;
            height: auto;
            max-width: 800px;
        }
        .color-block {
            display: flex;
            flex-direction: column;
            align-items: center;
            margin: 0 10px;
        }
        .color-box {
            align-items: center;
            width: 100px;
            height: 100px;
            border-radius: 5px;
            box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.2);
        }
        .color-label {
            margin-top: 5px;
            font-size: 12px;
            text-align: center;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.title("Dominant Color Picker")
    st.write("Upload an image to get its 5 dominant colors with Hex and RGB Code")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = np.array(cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Convert image to base64 for displaying with border
        _, buffer = cv2.imencode('.png', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        img_str = base64.b64encode(buffer).decode()

        st.markdown(f"<img class='uploadedImage' src='data:image/png;base64,{img_str}'/>", unsafe_allow_html=True)

        st.write("Generating palette...")
        ordered_colors, hex_colors = get_palette(image)

        st.write("Palette:")
        
        cols = st.columns(5)  # Create columns for better layout
        for i, color in enumerate(ordered_colors):
            rgb_color = tuple(map(int, color))
            hex_color = hex_colors[i]
            with cols[i]:
                st.markdown(
                    f'''
                    <div class="color-block">
                        <div class="color-box" style="background-color: {hex_color};"></div>
                        <div class="color-label">
                            RGB: {rgb_color}<br>HEX: {hex_color}
                        </div>
                    </div>
                    ''',
                    unsafe_allow_html=True
                )

if __name__ == "__main__":
    main()
