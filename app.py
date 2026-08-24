import streamlit as st
import numpy as np
from PIL import Image

# --- PIPELINE STAGES ---

# Stage 2: Image to Binary Matrix Conversion
class ImageToMatrixConverter:
    def __init__(self, target_size=(64, 64), threshold=128):
        self.target_size = target_size
        self.threshold = threshold

    def convert(self, pil_image: Image.Image) -> np.ndarray:
        img = pil_image.convert('L').resize(self.target_size)
        return (np.array(img) < self.threshold).astype(int)

# Stage 3: 4x4 Block Matrix Compression
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

# Stage 4: Density Profiling
class MatrixToRowColConverter:
    def profile(self, compressed_matrix: np.ndarray):
        return np.sum(compressed_matrix, axis=1), np.sum(compressed_matrix, axis=0)

# Stage 5: 16-Character Fingerprint Encoding
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

# Stage 6: LCS Comparison
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

# --- WEB UI FRONTEND ---

st.set_page_config(page_title="Signature Verification AI", layout="centered")

st.title("🖋️ Signature Verification System")
st.write("Upload two handwritten signatures to extract 16-character fingerprints and calculate LCS similarity.")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Signature 1 (Reference)")
    file1 = st.file_uploader("Upload Image 1", type=["png", "jpg", "jpeg"], key="img1")

with col2:
    st.subheader("Signature 2 (Test)")
    file2 = st.file_uploader("Upload Image 2", type=["png", "jpg", "jpeg"], key="img2")

if file1 and file2:
    img1 = Image.open(file1)
    img2 = Image.open(file2)

    with col1:
        st.image(img1, caption="Signature 1", use_container_width=True)
    with col2:
        st.image(img2, caption="Signature 2", use_container_width=True)

    if st.button("Compare Signatures", type="primary"):
        # Initialize pipeline components
        converter = ImageToMatrixConverter()
        compressor = MatrixCompressor()
        profiler = MatrixToRowColConverter()
        encoder = RowColCompressor()
        comparator = LCSComparator()

        # Process Image 1
        mat1 = converter.convert(img1)
        comp1 = compressor.compress(mat1)
        r1, c1 = profiler.profile(comp1)
        fp1 = encoder.compress_to_string(r1, c1)

        # Process Image 2
        mat2 = converter.convert(img2)
        comp2 = compressor.compress(mat2)
        r2, c2 = profiler.profile(comp2)
        fp2 = encoder.compress_to_string(r2, c2)

        # LCS Matching
        lcs_len, sim_score = comparator.compute_lcs(fp1, fp2)

        st.divider()
        st.header("Results")
        
        res_col1, res_col2 = st.columns(2)
        with res_col1:
            st.text(f"Fingerprint 1: {fp1}")
            st.text(f"Fingerprint 2: {fp2}")
        with res_col2:
            st.metric(label="LCS Match Length", value=f"{lcs_len} / 16")
            st.metric(label="Similarity Score", value=f"{sim_score:.1f}%")

        if sim_score >= 75.0:
            st.success("Verdict: High Structural Match (Signatures Likely Match)")
        elif sim_score >= 50.0:
            st.warning("Verdict: Moderate Match (Manual Verification Recommended)")
        else:
            st.error("Verdict: Low Match (Possible Forgery or Variant)")
