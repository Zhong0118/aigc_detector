import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import plotly.graph_objects as go
import requests

API_BASE = "http://localhost:8000/api/v1"

st.set_page_config(page_title="AIGC Detection Dashboard", layout="wide")
st.title("AIGC Detection & Attribution Dashboard")

tab1, tab2 = st.tabs(["Detection", "Report"])

with tab1:
    st.header("Upload Content for Detection")
    uploaded_file = st.file_uploader(
        "Choose a file", type=["txt", "md", "jpg", "jpeg", "png", "wav", "mp3", "mp4"]
    )

    if uploaded_file and st.button("Analyze"):
        with st.spinner("Detecting..."):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
            resp = requests.post(f"{API_BASE}/detect", files=files)

            if resp.status_code == 200:
                data = resp.json()

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Modality", data["modality"])
                with col2:
                    score = data["detection"]["score"]
                    st.metric("AI Score", f"{score:.2%}")
                with col3:
                    st.metric("Label", data["detection"]["label"].upper())

                if data["detection"]["modality_scores"]:
                    fig = go.Figure(data=[
                        go.Bar(
                            x=list(data["detection"]["modality_scores"].keys()),
                            y=list(data["detection"]["modality_scores"].values()),
                            marker_color=["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4"],
                        )
                    ])
                    fig.update_layout(
                        title="Modality Scores",
                        yaxis_title="AI Probability",
                        yaxis_range=[0, 1],
                    )
                    st.plotly_chart(fig, use_container_width=True)

                with st.expander("Details"):
                    st.json(data)
            else:
                st.error(f"Error: {resp.text}")

with tab2:
    st.header("View Report")
    content_id = st.number_input("Content ID", min_value=1, step=1)

    if st.button("Load Report"):
        resp = requests.get(f"{API_BASE}/report/{content_id}")
        if resp.status_code == 200:
            data = resp.json()
            st.subheader("Content Info")
            st.json(data["content"])

            if data["detection_results"]:
                st.subheader("Detection Results")
                for r in data["detection_results"]:
                    st.metric("Score", f"{r['score']:.2%}")
                    st.write(f"Label: **{r['label'].upper()}**")

            if data["provenance"]:
                st.subheader("Provenance")
                for p in data["provenance"]:
                    if p["attribution"]:
                        st.write("**Top-K Model Attribution:**")
                        for entry in p["attribution"]:
                            st.write(f"- {entry['model']}: {entry['probability']:.2%}")
        else:
            st.error("Report not found")
