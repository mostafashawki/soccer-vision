"""Streamlit UI for soccer-vision pipeline.

Provides a browser-based interface for:
- Video upload
- Processing configuration
- Real-time progress display
- Results visualization (annotated video, charts, report download)
"""

import json
import os
import time

import requests
import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates
import cv2
import numpy as np
import tempfile

# FastAPI backend URL (configurable via env)
API_URL = os.environ.get("API_URL", "http://localhost:8004")

st.set_page_config(
    page_title="Soccer Vision",
    page_icon="⚽",
    layout="wide",
)


def main():
    """Main Streamlit application."""
    if "frames_bgr" not in st.session_state:
        st.session_state.frames_bgr = []
    if "frames_rgb" not in st.session_state:
        st.session_state.frames_rgb = []
    if "current_frame_idx" not in st.session_state:
        st.session_state.current_frame_idx = 0
    if "clicks" not in st.session_state:
        st.session_state.clicks = {"team_a": [], "team_b": [], "other": []}
    if "video_name" not in st.session_state:
        st.session_state.video_name = ""
    if "last_click" not in st.session_state:
        st.session_state.last_click = None

    # Header
    st.title("⚽ Soccer Vision")
    st.markdown("**AI-powered player counting per team from soccer match footage**")
    st.divider()

    # Check API health
    api_healthy = False
    try:
        resp = requests.get(f"{API_URL}/health", timeout=5)
        api_healthy = resp.status_code == 200
    except requests.exceptions.ConnectionError:
        pass

    if not api_healthy:
        st.error("🔴 API server is not reachable. Make sure the API is running.")
        st.code(f"Expected API at: {API_URL}")
        st.stop()

    st.success("🟢 API server connected")

    # Sidebar — Configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        frame_skip = st.slider(
            "Frame Skip",
            min_value=1,
            max_value=30,
            value=1,
            help="Process every Nth frame. Higher = faster but less granular.",
        )
        save_video = st.checkbox("Generate annotated video", value=True)
        st.divider()
        st.markdown(
            "**soccer-vision** v1.0.0\n\n"
            "Upload a soccer match video to analyze "
            "player count per team across all frames."
        )

    # File upload
    col1, col2 = st.columns([2, 1])

    with col1:
        uploaded_file = st.file_uploader(
            "Upload a soccer video",
            type=["mp4", "avi", "mov"],
            help="Supported formats: MP4, AVI, MOV",
        )

    with col2:
        if uploaded_file:
            st.metric("File", uploaded_file.name)
            size_mb = uploaded_file.size / (1024 * 1024)
            st.metric("Size", f"{size_mb:.1f} MB")

    if not uploaded_file:
        st.info("👆 Upload a soccer match video to get started.")
        st.stop()

    # Extract 5 frames once per video
    if uploaded_file.name != st.session_state.video_name:
        st.session_state.video_name = uploaded_file.name
        st.session_state.frames_bgr = []
        st.session_state.frames_rgb = []
        st.session_state.clicks = {"team_a": [], "team_b": [], "other": []}
        st.session_state.abs_frame_indices = []
        st.session_state.current_frame_idx = 0
        st.session_state.last_click = None
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tfile:
            tfile.write(uploaded_file.getvalue())
            tfile_name = tfile.name
            
        cap = cv2.VideoCapture(tfile_name)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames > 0:
            frame_indices = [int(total_frames * idx) for idx in [0.05, 0.25, 0.5, 0.75, 0.95]]
            st.session_state.abs_frame_indices = frame_indices
            for f_idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, min(f_idx, total_frames - 1))
                ret, frame = cap.read()
                if ret:
                    st.session_state.frames_bgr.append(frame)
                    st.session_state.frames_rgb.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        os.remove(tfile_name)

    # Labeling UI
    if st.session_state.frames_rgb:
        st.divider()
        st.subheader("🎨 Multi-Frame Team Labeling (KNN)")
        st.markdown("Label a few players from each team across different frames to automatically train the classifier accurately handling different shirt angles.")
        
        # Controls
        col_ctrls, col_info = st.columns([1, 1])
        with col_ctrls:
            active_team = st.radio(
                "Selecting for:", 
                ["team_a", "team_b", "other"], 
                horizontal=True, 
                format_func=lambda x: {"team_a": "🔴 Team A", "team_b": "🔵 Team B", "other": "⚪ Ref/Gk"}[x]
            )
            
            c1, c2, c3 = st.columns([1, 1, 1])
            with c1:
                if st.button("⬅️ Prev Frame", disabled=(st.session_state.current_frame_idx == 0)):
                    st.session_state.current_frame_idx -= 1
                    st.session_state.last_click = None
                    st.rerun()
            with c2:
                st.write(f"Frame {st.session_state.current_frame_idx + 1} / {len(st.session_state.frames_rgb)}")
            with c3:
                if st.button("Next Frame ➡️", disabled=(st.session_state.current_frame_idx == len(st.session_state.frames_rgb)-1)):
                    st.session_state.current_frame_idx += 1
                    st.session_state.last_click = None
                    st.rerun()

        with col_info:
            c_a = len(st.session_state.clicks['team_a'])
            c_b = len(st.session_state.clicks['team_b'])
            c_o = len(st.session_state.clicks['other'])
            st.write(f"**Recorded:** Team A: {c_a} | Team B: {c_b} | Other: {c_o}")
            if st.button("🔄 Clear All Labels"):
                st.session_state.clicks = {"team_a": [], "team_b": [], "other": []}
                st.session_state.last_click = None
                st.rerun()
                
        # Image rendering with circles
        f_idx = st.session_state.current_frame_idx
        display_img = st.session_state.frames_rgb[f_idx].copy()
        for t, points in st.session_state.clicks.items():
            color = (255, 0, 0) if t == "team_a" else (0, 0, 255) if t == "team_b" else (200, 200, 200)
            for pt in points:
                if pt["frame"] == f_idx:
                    cv2.circle(display_img, (pt["x"], pt["y"]), max(15, display_img.shape[1] // 80), color, -1)
                    
        value = streamlit_image_coordinates(display_img, key=f"img_click_{f_idx}")
        
        if value is not None and value != st.session_state.last_click:
            st.session_state.last_click = value
            x, y = value["x"], value["y"]
            abs_frame_idx = st.session_state.abs_frame_indices[f_idx]
            
            st.session_state.clicks[active_team].append({"frame": f_idx, "abs_frame_idx": abs_frame_idx, "x": x, "y": y})
            st.rerun()
            st.rerun()

    # Process button
    st.divider()
    if st.button("🚀 Analyze Video", type="primary", use_container_width=True):
        with st.spinner("Processing video..."):
            progress_bar = st.progress(0, text="Uploading video...")
            
            try:
                # Upload and process
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "video/mp4")}
                params = {
                    "frame_skip": frame_skip, 
                    "save_video": save_video
                }
                
                train_data = {
                    "team_a": [{"frame": pt["abs_frame_idx"], "x": pt["x"], "y": pt["y"]} for pt in st.session_state.clicks["team_a"]],
                    "team_b": [{"frame": pt["abs_frame_idx"], "x": pt["x"], "y": pt["y"]} for pt in st.session_state.clicks["team_b"]],
                    "other": [{"frame": pt["abs_frame_idx"], "x": pt["x"], "y": pt["y"]} for pt in st.session_state.clicks["other"]]
                }
                
                form_data = None
                has_train_data = sum(len(x) for x in train_data.values()) > 0
                if has_train_data:
                    form_data = {"train_data_json": json.dumps(train_data)}

                progress_bar.progress(10, text="Processing... This may take a few minutes.")

                response = requests.post(
                    f"{API_URL}/process",
                    files=files,
                    params=params,
                    data=form_data,
                    timeout=600,  # 10 min timeout for long videos
                )

                if response.status_code != 200:
                    st.error(f"❌ Processing failed: {response.json().get('detail', 'Unknown error')}")
                    st.stop()

                result = response.json()
                job_id = result["job_id"]

                progress_bar.progress(100, text="✅ Processing complete!")

                st.session_state["job_id"] = job_id
                st.session_state["result"] = result

            except requests.exceptions.Timeout:
                st.error("❌ Processing timed out. Try with a shorter video or higher frame skip.")
                st.stop()
            except Exception as e:
                st.error(f"❌ Error: {e}")
                st.stop()

    # Display results
    if "job_id" in st.session_state:
        job_id = st.session_state["job_id"]
        st.divider()
        st.header("📊 Results")

        # Fetch report
        try:
            report_resp = requests.get(f"{API_URL}/results/{job_id}/report", timeout=30)
            if report_resp.status_code == 200:
                report = report_resp.json()

                # Summary metrics
                summary = report.get("summary", {})
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Avg Team A", f"{summary.get('avg_team_a', 0):.1f}")
                col2.metric("Avg Team B", f"{summary.get('avg_team_b', 0):.1f}")
                col3.metric("Avg Confidence", f"{summary.get('avg_confidence', 0):.0%}")
                col4.metric("Frames Analyzed", len(report.get("frames", [])))

                # Charts
                frames = report.get("frames", [])
                if frames:
                    st.subheader("📈 Player Count Over Time")

                    import pandas as pd

                    df = pd.DataFrame(frames)

                    # Line chart for team counts
                    chart_data = df[["timestamp_sec", "team_a_count", "team_b_count"]].set_index("timestamp_sec")
                    chart_data.columns = ["Team A", "Team B"]
                    st.line_chart(chart_data, use_container_width=True)

                    # Confidence chart
                    st.subheader("🎯 Detection Confidence")
                    conf_data = df[["timestamp_sec", "confidence"]].set_index("timestamp_sec")
                    conf_data.columns = ["Confidence"]
                    st.area_chart(conf_data, use_container_width=True)

                # Download report
                st.subheader("📥 Download")
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.download_button(
                        "📄 Download JSON Report",
                        data=json.dumps(report, indent=2),
                        file_name=f"{job_id}_report.json",
                        mime="application/json",
                    )

                # Try to get summary
                with col2:
                    try:
                        summary_resp = requests.get(f"{API_URL}/results/{job_id}/summary", timeout=10)
                        if summary_resp.status_code == 200:
                            summary_text = summary_resp.json().get("summary", "")
                            st.download_button(
                                "📝 Download Summary",
                                data=summary_text,
                                file_name=f"{job_id}_summary.txt",
                                mime="text/plain",
                            )
                    except Exception:
                        pass

                # Try to show annotated video
                if save_video:
                    with col3:
                        try:
                            video_resp = requests.get(f"{API_URL}/results/{job_id}/video", timeout=30)
                            if video_resp.status_code == 200:
                                st.download_button(
                                    "🎬 Download Video",
                                    data=video_resp.content,
                                    file_name=f"{job_id}_output.mp4",
                                    mime="video/mp4",
                                )
                        except Exception:
                            pass

                    # Show video player
                    try:
                        video_resp = requests.get(f"{API_URL}/results/{job_id}/video", timeout=30)
                        if video_resp.status_code == 200:
                            st.subheader("🎬 Annotated Video")
                            st.video(video_resp.content)
                    except Exception:
                        st.info("Annotated video not available for preview.")

        except Exception as e:
            st.error(f"Failed to fetch results: {e}")


if __name__ == "__main__":
    main()
