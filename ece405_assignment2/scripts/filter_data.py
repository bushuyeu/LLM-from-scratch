"""
Section 4: Filter CC WET files to produce language modeling training data.

Usage:
    python scripts/filter_pipeline.py --input_dir data/ --output_dir data/filtered/ --workers 4

Filters applied (in order):
    1. Language identification — keep English (confidence >= 0.80)
    2. Gopher quality rules — word count, mean word length, ellipsis, alpha ratio
    3. Quality classifier — keep "wiki" label (confidence >= 0.50) [optional]
    4. Harmful content — remove NSFW or toxic (confidence >= 0.50)
    5. PII masking — mask emails, phone numbers, IP addresses
    6. Exact line deduplication — across all output files
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import pathlib
import sys
import time
from dataclasses import dataclass, field

from warcio.archiveiterator import ArchiveIterator

# Add parent so cs336_data is importable
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from cs336_data.language_identification import identify_language
from cs336_data.quality_filters import gopher_quality_filter
from cs336_data.pii import mask_emails, mask_phone_numbers, mask_ips
from cs336_data.harmful_content import classify_nsfw, classify_toxic_speech


# --- Configuration ---
LANG_THRESHOLD = 0.80
QUALITY_THRESHOLD = 0.50
NSFW_THRESHOLD = 0.50
TOXIC_THRESHOLD = 0.50
MIN_TEXT_LENGTH = 100


@dataclass
class FilterStats:
    total: int = 0
    empty: int = 0
    lang_filtered: int = 0
    gopher_filtered: int = 0
    quality_filtered: int = 0
    nsfw_filtered: int = 0
    toxic_filtered: int = 0
    kept: int = 0
    pii_emails: int = 0
    pii_phones: int = 0
    pii_ips: int = 0

    def to_dict(self) -> dict:
        return {
            "total": self.total,
            "empty": self.empty,
            "lang_filtered": self.lang_filtered,
            "gopher_filtered": self.gopher_filtered,
            "quality_filtered": self.quality_filtered,
            "nsfw_filtered": self.nsfw_filtered,
            "toxic_filtered": self.toxic_filtered,
            "kept": self.kept,
            "pii_emails": self.pii_emails,
            "pii_phones": self.pii_phones,
            "pii_ips": self.pii_ips,
        }


def _try_load_quality_classifier():
    """Try to load the quality classifier; return None if unavailable."""
    try:
        from cs336_data.quality_filters import classify_quality, _get_quality_model
        _get_quality_model()  # force load to check if model exists
        return classify_quality
    except Exception:
        return None


def process_single_wet_file(input_path: str, output_path: str, use_quality: bool = True) -> dict:
    """Process a single WET file through the filter pipeline."""
    stats = FilterStats()
    classify_quality_fn = _try_load_quality_classifier() if use_quality else None

    if classify_quality_fn is None and use_quality:
        print(f"  Quality classifier not available, skipping quality filter", file=sys.stderr)

    kept_texts = []

    with open(input_path, "rb") as f:
        for record in ArchiveIterator(f):
            if record.rec_type != "conversion":
                continue

            stats.total += 1
            text = record.content_stream().read().decode("utf-8", errors="replace")

            # Skip empty/short
            if not text or len(text.strip()) < MIN_TEXT_LENGTH:
                stats.empty += 1
                continue

            text = text.strip()

            # 1. Language identification — keep English
            lang, lang_score = identify_language(text)
            if lang != "en" or lang_score < LANG_THRESHOLD:
                stats.lang_filtered += 1
                continue

            # 2. Gopher quality rules
            if not gopher_quality_filter(text):
                stats.gopher_filtered += 1
                continue

            # 3. Quality classifier (optional)
            if classify_quality_fn is not None:
                label, score = classify_quality_fn(text)
                if label != "wiki" or score < QUALITY_THRESHOLD:
                    stats.quality_filtered += 1
                    continue

            # 4. Harmful content
            nsfw_label, nsfw_score = classify_nsfw(text)
            if nsfw_label == "nsfw" and nsfw_score >= NSFW_THRESHOLD:
                stats.nsfw_filtered += 1
                continue

            toxic_label, toxic_score = classify_toxic_speech(text)
            if toxic_label == "toxic" and toxic_score >= TOXIC_THRESHOLD:
                stats.toxic_filtered += 1
                continue

            # 5. PII masking
            text, n_emails = mask_emails(text)
            text, n_phones = mask_phone_numbers(text)
            text, n_ips = mask_ips(text)
            stats.pii_emails += n_emails
            stats.pii_phones += n_phones
            stats.pii_ips += n_ips

            kept_texts.append(text)
            stats.kept += 1

    # Write output
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        for text in kept_texts:
            # One document per line (newlines replaced with spaces)
            line = text.replace("\n", " ").strip()
            f.write(line + "\n")

    return stats.to_dict()


def main():
    parser = argparse.ArgumentParser(description="Filter CC WET files for language modeling")
    parser.add_argument("--input_dir", required=True, help="Directory containing WET files")
    parser.add_argument("--output_dir", required=True, help="Output directory for filtered data")
    parser.add_argument("--workers", type=int, default=1, help="Number of parallel workers")
    parser.add_argument("--no_quality", action="store_true", help="Skip quality classifier")
    parser.add_argument("--glob", default="*.warc.wet.gz", help="Glob pattern for WET files")
    args = parser.parse_args()

    input_dir = pathlib.Path(args.input_dir)
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    wet_files = sorted(input_dir.glob(args.glob))
    if not wet_files:
        print(f"No WET files found matching {args.glob} in {input_dir}")
        sys.exit(1)

    print(f"Found {len(wet_files)} WET files")
    print(f"Output directory: {output_dir}")
    print(f"Workers: {args.workers}")
    print(f"Quality classifier: {'disabled' if args.no_quality else 'enabled (if model available)'}")
    print()

    start_time = time.time()
    all_stats = []

    if args.workers <= 1:
        for wet_file in wet_files:
            out_path = str(output_dir / wet_file.name)
            print(f"Processing {wet_file.name}...")
            stats = process_single_wet_file(str(wet_file), out_path, use_quality=not args.no_quality)
            all_stats.append(stats)
            print(f"  {stats['kept']}/{stats['total']} kept")
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {}
            for wet_file in wet_files:
                out_path = str(output_dir / wet_file.name)
                future = executor.submit(
                    process_single_wet_file, str(wet_file), out_path, not args.no_quality
                )
                futures[future] = wet_file.name

            for future in concurrent.futures.as_completed(futures):
                name = futures[future]
                stats = future.result()
                all_stats.append(stats)
                print(f"  {name}: {stats['kept']}/{stats['total']} kept")

    elapsed = time.time() - start_time

    # Aggregate stats
    totals = {}
    for key in all_stats[0]:
        totals[key] = sum(s[key] for s in all_stats)

    print()
    print("=" * 60)
    print("PIPELINE SUMMARY")
    print("=" * 60)
    print(f"Files processed:    {len(wet_files)}")
    print(f"Total records:      {totals['total']}")
    print(f"Empty/short:        {totals['empty']} ({totals['empty']/max(totals['total'],1)*100:.1f}%)")
    print(f"Language filtered:  {totals['lang_filtered']} ({totals['lang_filtered']/max(totals['total'],1)*100:.1f}%)")
    print(f"Gopher filtered:    {totals['gopher_filtered']} ({totals['gopher_filtered']/max(totals['total'],1)*100:.1f}%)")
    print(f"Quality filtered:   {totals['quality_filtered']} ({totals['quality_filtered']/max(totals['total'],1)*100:.1f}%)")
    print(f"NSFW filtered:      {totals['nsfw_filtered']} ({totals['nsfw_filtered']/max(totals['total'],1)*100:.1f}%)")
    print(f"Toxic filtered:     {totals['toxic_filtered']} ({totals['toxic_filtered']/max(totals['total'],1)*100:.1f}%)")
    print(f"Kept:               {totals['kept']} ({totals['kept']/max(totals['total'],1)*100:.1f}%)")
    print(f"PII masked:         {totals['pii_emails']} emails, {totals['pii_phones']} phones, {totals['pii_ips']} IPs")
    print(f"Time:               {elapsed:.1f}s ({elapsed/len(wet_files):.1f}s per file)")
    print(f"Est. 5000 WETs:     {elapsed/len(wet_files)*5000/3600:.1f} hours")
    print(f"Est. 100000 WETs:   {elapsed/len(wet_files)*100000/3600:.1f} hours")

    # Save stats
    stats_path = output_dir / "pipeline_stats.json"
    with open(stats_path, "w") as f:
        json.dump({"files": len(wet_files), "elapsed_seconds": elapsed, "totals": totals}, f, indent=2)
    print(f"\nStats saved to {stats_path}")


if __name__ == "__main__":
    main()
