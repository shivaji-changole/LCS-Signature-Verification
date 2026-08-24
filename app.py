from flask import Flask, render_template, request, jsonify
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

# --- PIPELINE STAGES ---

# Stage 2: Convert to 64x64 binary matrix
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

# Pipeline Orchestrator
def process_images(file1, file2):
    converter = ImageToMatrixConverter()
    compressor = MatrixCompressor()
    profiler = MatrixToRowColConverter()
    encoder = RowColCompressor()
    comparator = LCSComparator()

    img1 = Image.open(io.BytesIO(file1.read()))
    img2 = Image.open(io.BytesIO(file2.read()))

    # Signature 1
    m1 = converter.convert(img1)
    c1 = compressor.compress(m1)
    r1_d, c1_d = profiler.profile(c1)
    fp1 = encoder.compress_to_string(r1_d, c1_d)

    # Signature 2
    m2 = converter.convert(img2)
    c2 = compressor.compress(m2)
    r2_d, c2_d = profiler.profile(c2)
    fp2 = encoder.compress_to_string(r2_d, c2_d)

    lcs_len, score = comparator.compute_lcs(fp1, fp2)

    return {
        "fp1": fp1,
        "fp2": fp2,
        "lcs_len": lcs_len,
        "similarity": round(score, 1)
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/compare', methods=['POST'])
def compare_api():
    if 'img1' not in request.files or 'img2' not in request.files:
        return jsonify({"error": "Both image files are required."}), 400

    file1 = request.files['img1']
    file2 = request.files['img2']

    results = process_images(file1, file2)
    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
