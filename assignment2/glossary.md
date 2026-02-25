# Glossary of Terms: CS336 Assignment 4 (Data)

This glossary contains key terms and definitions from the Data Assignment handout, arranged chronologically as they appear in the document.

## 1. Assignment Overview
*   **Common Crawl (CC)**: A non-profit that provides a free corpus of web pages by crawling the internet.
*   **WET (Web Extracted Text)**: A Common Crawl file format containing extracted plain text from raw HTML pages.
*   **WARC (Web ARChive format)**: A Common Crawl file format containing raw request/response data, including raw HTML and HTTP headers.
*   **WAT (Web Archive Transformation)**: A Common Crawl file format containing higher-level metadata extracted from WARC files, stored as JSON objects.
*   **Deduplication**: The process of removing duplicate or near-duplicate content from a dataset to improve training efficiency and model quality.

## 2. Filtering Common Crawl
*   **Resiliparse**: A library used for efficient text extraction from HTML and encoding detection.
*   **FastWARC**: A Python library used for iterating over and processing WARC, WAT, and WET files.
*   **Language Identification**: The process of detecting the language of a document. Often performed using tools like **fastText**.
*   **Personal Identifiable Information (PII)**: Sensitive information used to identify individuals, such as email addresses, phone numbers, and IP addresses.
*   **NSFW (Not Safe For Work)**: A category of content including pornography, profanity, or other potentially disturbing material.
*   **Toxic Speech**: Language that is rude, disrespectful, or unreasonable and likely to make someone leave a discussion.
*   **Gopher Quality Filters**: A set of heuristic rules (e.g., word count, word length, character distribution) used to remove low-quality text.
*   **Quality Classifier**: A machine learning model (e.g., fastText) trained to distinguish between high-quality and low-quality text, often using reference datasets like Wikipedia or Reddit.

## 3. Deduplication
*   **Exact Line Deduplication**: A method that removes lines that are repeated verbatim across multiple documents in a corpus.
*   **Fuzzy Deduplication**: Identifying and removing "mostly-templated" or near-duplicate documents that aren't exact matches.
*   **Jaccard Similarity**: A metric for similarity between two sets $S$ and $T$, defined as $|S \cap T| / |S \cup T|$.
*   **N-grams**: Contiguous sequences of $n$ words or characters from a given text, used to represent document content for similarity comparisons.
*   **MinHashing**: A memory-efficient technique to estimate Jaccard similarity between documents by creating small "signatures" from their n-gram sets.
*   **Locality-Sensitive Hashing (LSH)**: An algorithm that buckets documents with similar signatures into candidate duplicate pairs, avoiding $O(n^2)$ comparisons.
*   **Bands and Rows (r and b)**: LSH parameters where the signature of length $k$ is divided into $b$ bands of $r$ rows ($k=br$). Increasing $b$ increases recall.

## 4. Leaderboard & Training
*   **Paloma Benchmark**: A benchmark for evaluating language model fit, often using specific subsets like the C4 100 domains.
*   **GPT-2 Tokenizer**: A specific byte-level BPE tokenizer used as a standard for many models.
*   **Slurm**: A cluster management and job scheduling system used to parallelize data processing and training across multiple machines.
*   **Submitit**: A Python library used to launch jobs on Slurm partitions via a simple interface.
*   **Distributed Data Parallel (DDP)**: A PyTorch feature used to parallelize training across multiple GPUs.
