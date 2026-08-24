import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image

# --- PIPELINE CLASSES ---

class ImageToMatrixConverter:
    def __init__(self, target_size=(64, 64), threshold=128):
        self.target_size = target_size
        self.threshold = threshold

    def convert(self, pil_image: Image.Image) -> np.ndarray:
        img = pil_image.convert('L').resize(self.target_size)
        return (np.array(img) < self.threshold).astype(int)

class MatrixCompressor:
    def __init__(self, block_size=(4, 4), ink_threshold_ratio=0.10):
        self.block_size = block_size
        self.ink_threshold_ratio = ink_threshold_ratio

    def compress(self, matrix: np.ndarray) -> np.ndarray:
        h, w = matrix.shape
        bh, bw = self.block_size
        out_h, out_w = h // bh, w // bw
        compressed = np.zeros((out_h, out_w), dtype=int)
        min_ink = (bh * bw) * self.ink_threshold_ratio

        for i in range(out_h):
            for j in range(out_w):
                if np.sum(matrix[i*bh:(i+1)*bh, j*bw:(j+1)*bw]) >= min_ink:
                    compressed[i, j] = 1
        return compressed

class MatrixToRowColConverter:
    def profile(self, compressed_matrix: np.ndarray):
        return np.sum(compressed_matrix, axis=1), np.sum(compressed_matrix, axis=0)

class RowColCompressor:
    def compress_to_string(self, row_density: np.ndarray, col_density: np.ndarray) -> str:
        row_buckets = [np.mean(row_density[i:i+2]) for i in range(0, 16, 2)]
        col_buckets = [np.mean(col_density[i:i+2]) for i in range(0, 16, 2)]
        combined = row_buckets + col_buckets
        max_val = max(combined) if max(combined) > 0 else 1.0
        hex_chars = "0123456789ABCDEF"

        fingerprint = []
        for val in combined:
            level = int(round((val / max_val) * 15))
            level = min(max(level, 0), 15)
            fingerprint.append(hex_chars[level])
        return "".join(fingerprint)

class LCSComparator:
    def compute_lcs(self, str1: str, str2: str):
        m, n = len(str1), len(str2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if str1[i - 1] == str2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

        lcs_length = dp[m][n]
        similarity = (lcs_length / max(min(m, n), 1)) * 100.0
        return lcs_length, similarity

# --- PAGE CONFIG & RESPONSIVE STYLING ---

st.set_page_config(
    page_title="SigVerify Pro | Enterprise Verification",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="auto"
)

# Responsive Enterprise CSS (Desktop + Mobile Support)
st.markdown("""
    <style>
    /* Desktop Base Layout */
    .main .block-container {
        padding-top: 1.8rem;
        padding-bottom: 3rem;
        max-width: 1100px;
    }
    
    .app-title {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        font-size: 2rem;
        font-weight: 700;
        color: #0F172A;
        margin-bottom: 0.2rem;
    }
    .app-subtitle {
        font-size: 0.95rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }

    /* Dynamic Responsive Fingerprint Display */
    .fp-box {
        background-color: #F8FAFC;
        border: 1px solid #CBD5E1;
        border-radius: 8px;
        padding: 10px 8px;
        font-family: 'Courier New', monospace;
        font-weight: 700;
        font-size: clamp(0.85rem, 3.5vw, 1.15rem);
        letter-spacing: clamp(1px, 0.4vw, 3px);
        color: #1E293B;
        text-align: center;
        word-break: break-all;
        box-shadow: inset 0 1px 2px rgba(0, 0, 0, 0.03);
    }
    
    /* Touch-Optimized Primary Action Buttons */
    .stButton>button {
        min-height: 48px;
        font-size: 1rem;
        font-weight: 600;
        border-radius: 8px;
    }

    /* Verdict Alert Boxes */
    .verdict-box {
        padding: 0.9rem 1.1rem;
        border-radius: 8px;
        font-weight: 600;
        font-size: 0.95rem;
        text-align: center;
        margin-top: 1rem;
        line-height: 1.4;
    }
    .verdict-pass {
        background-color: #F0FDF4;
        border: 1px solid #BBF7D0;
        color: #166534;
    }
    .verdict-warn {
        background-color: #FEFCE8;
        border: 1px solid #FEF08A;
        color: #854D0E;
    }
    .verdict-fail {
        background-color: #FEF2F2;
        border: 1px solid #FECACA;
        color: #991B1B;
    }

    /* --- MOBILE MEDIA QUERIES (<768px) --- */
    @media (max-width: 768px) {
        .main .block-container {
            padding-top: 1rem !important;
            padding-bottom: 2rem !important;
            padding-left: 0.75rem !important;
            padding-right: 0.75rem !important;
        }
        .app-title {
            font-size: 1.5rem !important;
        }
        .app-subtitle {
            font-size: 0.85rem !important;
            margin-bottom: 1rem !important;
        }
        /* Optimize metrics display on mobile */
        [data-testid="stMetricValue"] {
            font-size: 1.25rem !important;
        }
        [data-testid="stMetricLabel"] {
            font-size: 0.8rem !important;
        }
        /* Mobile tab padding adjustments */
        .stTabs [data-baseweb="tab"] {
            padding: 8px 12px !important;
            font-size: 0.85rem !important;
        }
    }
    </style>
""", unsafe_allow_html=True)

# --- SIDEBAR CONFIGURATION ---

with st.sidebar:
    st.image("https://img.icons8.com/isometric-line/100/4a6cf7/shield.png", width=44)
    st.title("Control Panel")
    st.caption("Adjust pipeline settings")
    
    st.markdown("**Image Preprocessing**")
    bin_threshold = st.slider("Binarization Threshold", 50, 200, 128, help="Adjust stroke detection threshold.")
    
    st.markdown("**Encoding Parameters**")
    ink_ratio = st.slider("Ink Threshold Ratio", 0.05, 0.30, 0.10, step=0.01)
    
    st.markdown("**Match Tolerances**")
    high_match_thresh = st.slider("High Match Threshold (%)", 60.0, 90.0, 75.0, step=5.0)
    mod_match_thresh = st.slider("Moderate Match Threshold (%)", 40.0, 70.0, 50.0, step=5.0)

# --- MAIN CONTENT LAYOUT ---

st.markdown('<div class="app-title">🛡️ SigVerify Pro</div>', unsafe_allow_html=True)
st.markdown('<div class="app-subtitle">Biometric Signature Verification • Density Encoding & LCS Matching</div>', unsafe_allow_html=True)

st.subheader("1. Signature Acquisition")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Reference Signature**")
    file1 = st.file_uploader("Upload Reference Signature", type=["png", "jpg", "jpeg"], key="img1", label_visibility="collapsed")
    if file1:
        img1 = Image.open(file1)
        st.image(img1, caption="Reference", use_container_width=True)

with col2:
    st.markdown("**Test Signature**")
    file2 = st.file_uploader("Upload Test Signature", type=["png", "jpg", "jpeg"], key="img2", label_visibility="collapsed")
    if file2:
        img2 = Image.open(file2)
        st.image(img2, caption="Test Sample", use_container_width=True)

st.divider()

# --- VERIFICATION EXECUTION ---

if file1 and file2:
    st.subheader("2. Authentication & Analysis")
    
    if st.button("Run Verification Analysis", type="primary", use_container_width=True):
        with st.spinner("Processing signatures..."):
            converter = ImageToMatrixConverter(threshold=bin_threshold)
            compressor = MatrixCompressor(ink_threshold_ratio=ink_ratio)
            profiler = MatrixToRowColConverter()
            encoder = RowColCompressor()
            comparator = LCSComparator()

            # Process Reference
            mat1 = converter.convert(img1)
            comp1 = compressor.compress(mat1)
            r1, c1 = profiler.profile(comp1)
            fp1 = encoder.compress_to_string(r1, c1)

            # Process Test Sample
            mat2 = converter.convert(img2)
            comp2 = compressor.compress(mat2)
            r2, c2 = profiler.profile(comp2)
            fp2 = encoder.compress_to_string(r2, c2)

            # LCS Matching
            lcs_len, sim_score = comparator.compute_lcs(fp1, fp2)

        tab_results, tab_forensics = st.tabs(["📊 Executive Summary", "🔬 Density Forensics"])

        with tab_results:
            # Responsive Metrics Row
            m_col1, m_col2, m_col3 = st.columns(3)
            m_col1.metric("LCS Match", f"{lcs_len} / 16")
            m_col2.metric("Similarity", f"{sim_score:.1f}%")
            
            if sim_score >= high_match_thresh:
                m_col3.metric("Status", "PASSED", delta="Authentic")
            elif sim_score >= mod_match_thresh:
                m_col3.metric("Status", "REVIEW", delta="Check Required", delta_color="off")
            else:
                m_col3.metric("Status", "FAILED", delta="High Risk", delta_color="inverse")

            st.progress(sim_score / 100.0)

            # Responsive Fingerprint Cards
            fp_col1, fp_col2 = st.columns(2)
            with fp_col1:
                st.markdown("**Reference Fingerprint**")
                st.markdown(f'<div class="fp-box">{fp1}</div>', unsafe_allow_html=True)
            with fp_col2:
                st.markdown("**Test Sample Fingerprint**")
                st.markdown(f'<div class="fp-box">{fp2}</div>', unsafe_allow_html=True)

            # Mobile-optimized Alert Box
            if sim_score >= high_match_thresh:
                st.markdown('<div class="verdict-box verdict-pass">✓ AUTHENTICATION SUCCESSFUL: High structural density match detected.</div>', unsafe_allow_html=True)
            elif sim_score >= mod_match_thresh:
                st.markdown('<div class="verdict-box verdict-warn">⚠️ MODERATE MATCH: Structural variations present. Secondary review recommended.</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="verdict-box verdict-fail">🚨 AUTHENTICATION FAILED: Significant density mismatch. Possible forgery.</div>', unsafe_allow_html=True)

        with tab_forensics:
            st.markdown("**Density Distributions**")
            chart_col1, chart_col2 = st.columns(2)
            
            with chart_col1:
                st.caption("Reference Signature Density Profile")
                df_ref = pd.DataFrame({"Row Density": r1, "Col Density": c1})
                st.bar_chart(df_ref)

            with chart_col2:
                st.caption("Test Signature Density Profile")
                df_test = pd.DataFrame({"Row Density": r2, "Col Density": c2})
                st.bar_chart(df_test)
else:
    st.info("💡 Upload reference and test signatures above to proceed.")
