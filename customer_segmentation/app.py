import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from model import load_data, perform_clustering

st.title("Mall Customer Segmentation 🚀")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    data = load_data(uploaded_file)
    st.write("### Raw Data", data)

    features = st.multiselect("Select Features for Clustering", data.columns)
    
    if features:
        n_clusters = st.slider("Number of Clusters", 2, 10, 3)
        clustered_data, model = perform_clustering(data[features], n_clusters)
        
        st.write("### Clustered Data", clustered_data)

        # Visualization
        fig, ax = plt.subplots()
        sns.scatterplot(
            x=features[0],
            y=features[1],
            hue=clustered_data['Cluster'],
            palette='Set1',
            data=clustered_data,
            ax=ax
        )
        plt.title('Customer Segments')
        st.pyplot(fig)
