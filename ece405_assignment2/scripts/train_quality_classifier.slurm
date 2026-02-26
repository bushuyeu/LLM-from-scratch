#!/bin/bash
#SBATCH --job-name=quality-clf
#SBATCH --partition=sandbox
#SBATCH --time=04:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --output=quality_classifier_%j.log
#SBATCH --error=quality_classifier_%j.err

set -euo pipefail

# --- Load modules ---
module load lang/Python/3.11.5-GCCcore-13.2.0
module load compiler/GCC/13.2.0

# --- Storage paths ---
SCRATCH="/mnt/lustre/koa/scratch/pavelb/quality_classifier"
DATA_DIR="${SCRATCH}/data"
VENV_DIR="/tmp/${USER}-quality-venv-${SLURM_JOB_ID}"
mkdir -p "${DATA_DIR}"

echo "=== Job ${SLURM_JOB_ID} started at $(date) on $(hostname) ==="
echo "Scratch: ${SCRATCH}"

# --- Set up Python venv ---
echo "=== Setting up Python venv ==="
python3 -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"
pip install --quiet fasttext warcio resiliparse nltk pybind11

# Download NLTK punkt tokenizer
python3 -c "import nltk; nltk.download('punkt_tab', quiet=True)"

# --- Step 1: Download Wikipedia URLs ---
WIKI_URLS="${DATA_DIR}/enwiki-20240420-extracted_urls.txt.gz"
if [ ! -f "${WIKI_URLS}" ]; then
    echo "=== Downloading Wikipedia URLs from Stanford ==="
    wget -q -O "${WIKI_URLS}" "https://nlp.stanford.edu/data/nfliu/cs336-spring-2024/assignment4/enwiki-20240420-extracted_urls.txt.gz"
fi
echo "Wikipedia URLs: $(zcat "${WIKI_URLS}" | wc -l) total"

# --- Step 2: Download CC WARC for negative examples ---
CC_WARC="${DATA_DIR}/example.warc.gz"
if [ ! -f "${CC_WARC}" ]; then
    echo "=== Downloading Common Crawl WARC ==="
    wget -q -O "${CC_WARC}" "https://data.commoncrawl.org/crawl-data/CC-MAIN-2025-18/segments/1744889135610.12/warc/CC-MAIN-20250417135010-20250417165010-00065.warc.gz"
fi
echo "CC WARC: $(ls -lh "${CC_WARC}" | awk '{print $5}')"

# --- Step 3: Subsample and scrape Wikipedia-linked URLs ---
N_URLS=20000
SAMPLED_URLS="${DATA_DIR}/sampled_wiki_urls.txt"
WIKI_WARC="${DATA_DIR}/wiki_positive.warc.gz"

if [ ! -f "${WIKI_WARC}" ]; then
    echo "=== Subsampling ${N_URLS} Wikipedia URLs ==="
    python3 -c "
import gzip, random
random.seed(42)
with gzip.open('${WIKI_URLS}', 'rt') as f:
    urls = [l.strip() for l in f if l.strip()]
sampled = random.sample(urls, min(${N_URLS}, len(urls)))
with open('${SAMPLED_URLS}', 'w') as f:
    f.writelines(u + '\n' for u in sampled)
print(f'Sampled {len(sampled)} URLs')
"

    echo "=== Scraping Wikipedia-linked pages with wget ==="
    wget --timeout=5 --tries=1 \
        -i "${SAMPLED_URLS}" \
        --warc-file="${DATA_DIR}/wiki_positive" \
        -O /dev/null \
        --no-check-certificate \
        -q 2>&1 || true
    echo "Scraping done. WARC: $(ls -lh "${WIKI_WARC}" | awk '{print $5}')"
fi

# --- Step 4: Extract text and build training data ---
echo "=== Extracting text and building training data ==="
TRAIN_FILE="${DATA_DIR}/quality_train.txt"

python3 << PYTHON_SCRIPT
import random
import sys
from warcio.archiveiterator import ArchiveIterator
from resiliparse.extract.html2text import extract_plain_text
from resiliparse.parse.encoding import detect_encoding
import nltk

try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)


def extract_text_from_html_bytes(html_bytes):
    try:
        html_str = html_bytes.decode("utf-8")
    except UnicodeDecodeError:
        encoding = detect_encoding(html_bytes)
        if encoding:
            try:
                html_str = html_bytes.decode(encoding)
            except (UnicodeDecodeError, LookupError):
                return None
        else:
            return None
    text = extract_plain_text(html_str)
    return text if text and text.strip() else None


def gopher_quality_filter(text):
    words = nltk.word_tokenize(text)
    if not words:
        return False
    alpha_words = [w for w in words if any(c.isalpha() for c in w)]
    num_words = len(alpha_words)
    if num_words < 50 or num_words > 100_000:
        return False
    mean_word_len = sum(len(w) for w in alpha_words) / num_words
    if mean_word_len < 3 or mean_word_len > 10:
        return False
    lines = text.split("\n")
    if lines:
        ellipsis_count = sum(1 for line in lines if line.rstrip().endswith("...") or line.rstrip().endswith("\u2026"))
        if ellipsis_count / len(lines) > 0.3:
            return False
    alpha_ratio = len(alpha_words) / len(words)
    if alpha_ratio < 0.8:
        return False
    return True


def extract_from_warc(warc_path, max_docs=None, min_length=200):
    texts = []
    try:
        with open(warc_path, "rb") as f:
            for record in ArchiveIterator(f):
                if record.rec_type == "response":
                    try:
                        html_bytes = record.content_stream().read()
                        text = extract_text_from_html_bytes(html_bytes)
                        if text and len(text.strip()) >= min_length:
                            texts.append(text.strip())
                    except Exception:
                        continue
                if max_docs and len(texts) >= max_docs:
                    break
    except Exception as e:
        print(f"  Warning: error reading {warc_path}: {e}", file=sys.stderr)
    return texts


DATA_DIR = "${DATA_DIR}"
TRAIN_FILE = "${TRAIN_FILE}"

# Extract positive examples
print("Extracting positive examples (Wikipedia-linked pages)...")
wiki_texts = extract_from_warc(f"{DATA_DIR}/wiki_positive.warc.gz")
print(f"  Raw: {len(wiki_texts)} documents")
wiki_filtered = [t for t in wiki_texts if gopher_quality_filter(t)]
print(f"  After Gopher filter: {len(wiki_filtered)}")

# Extract negative examples
print("Extracting negative examples (Common Crawl)...")
cc_texts = extract_from_warc(f"{DATA_DIR}/example.warc.gz", max_docs=10000)
print(f"  Raw: {len(cc_texts)} documents")
cc_filtered = [t for t in cc_texts if gopher_quality_filter(t)]
print(f"  After Gopher filter: {len(cc_filtered)}")

# Balance and write training file
n_examples = min(len(wiki_filtered), len(cc_filtered))
if n_examples == 0:
    print("ERROR: No examples after filtering!", file=sys.stderr)
    sys.exit(1)

random.seed(42)
wiki_train = random.sample(wiki_filtered, n_examples)
cc_train = random.sample(cc_filtered, n_examples)
print(f"Balanced: {n_examples} examples per class")

with open(TRAIN_FILE, "w") as f:
    for text in wiki_train:
        line = text.replace("\n", " ").strip()
        f.write(f"__label__wiki {line}\n")
    for text in cc_train:
        line = text.replace("\n", " ").strip()
        f.write(f"__label__cc {line}\n")

print(f"Training file written: {TRAIN_FILE} ({n_examples * 2} lines)")
PYTHON_SCRIPT

# --- Step 5: Train fastText classifier ---
echo "=== Training fastText classifier ==="
MODEL_PATH="${SCRATCH}/quality_classifier.bin"

python3 << TRAIN_SCRIPT
import fasttext

model = fasttext.train_supervised(
    input="${TRAIN_FILE}",
    lr=0.1,
    epoch=25,
    wordNgrams=2,
    dim=100,
    loss="softmax",
)
model.save_model("${MODEL_PATH}")
print(f"Model saved to ${MODEL_PATH}")
print(f"Labels: {model.labels}")

# Quick test
result = model.test("${TRAIN_FILE}")
print(f"Training accuracy: {result[1]:.4f} (P), {result[2]:.4f} (R)")

# Test on sample texts
print()
print("Sample predictions:")
print("  Wiki-like:", model.predict("Anarchism is a political theory that is skeptical of the justification of authority and power."))
print("  CC-like:", model.predict("Buy cheap viagra online now! Best prices guaranteed!!!"))
TRAIN_SCRIPT

echo "=== Job completed at $(date) ==="
echo "Model saved to: ${MODEL_PATH}"

# Cleanup
deactivate
rm -rf "${VENV_DIR}"
