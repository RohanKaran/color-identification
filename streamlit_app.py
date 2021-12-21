import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import cv2
from collections import Counter
# from skimage.color import rgb2lab, deltaE_cie76
import os
import streamlit as st


def RGB2HEX(color):
    return "#{:02x}{:02x}{:02x}".format(int(color[0]), int(color[1]), int(color[2]))


def get_colors(image, mc):
    length = int(600*float(image.shape[0] / image.shape[1]))
    image = cv2.resize(image, (600, length), interpolation=cv2.INTER_AREA)
    # st.image(image)
    modified_image = image.reshape(image.shape[0] * image.shape[1], 3)
    clf = KMeans(n_clusters=mc, random_state=42)
    labels = clf.fit_predict(modified_image)

    counts = Counter(labels)

    center_colors = clf.cluster_centers_
    # We get ordered colors by iterating through the keys
    ordered_colors = [center_colors[i] for i in counts.keys()]
    hex_colors = [RGB2HEX(ordered_colors[i]) for i in counts.keys()]
    rgb_colors = [ordered_colors[i] for i in counts.keys()]

    # plt.figure(figsize=(12, 8))
    fig1, ax1 = plt.subplots()

    def make_autopct():
        def my_autopct(pct):
            return '{p:.2f}%'.format(p=pct)

        return my_autopct

    ax1.pie(counts.values(), labels=hex_colors, colors=hex_colors, autopct=make_autopct(),
            rotatelabels=True, wedgeprops={'animated': True}, textprops={'size': 5, 'color': "black"})
    fig1.patch.set_facecolor(color="None")
    st.subheader("Colors:")
    st.pyplot(fig1)

    return rgb_colors


if __name__ == "__main__":
    st.title("Color Identification in Images")

    uploaded_file = st.file_uploader("Choose a image file", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        up_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
        st.subheader("Uploaded image:")
        st.image(up_image)
        st.sidebar.subheader("Maximum colors:")
        max_colors = st.sidebar.slider('Choose between 1-10', min_value=1, max_value=10, value=5)

        with st.spinner("Analyzing..."):
            get_colors(up_image, max_colors)
            st.success("Done!")
