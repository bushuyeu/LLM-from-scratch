# UHM ECE 405 Spring 2026 Assignment 2: Data

This asignment is created from Assignment 4 of [CS336 at Stanford taught in Spring 2025](https://stanford-cs336.github.io/spring2025/). 
For the full description of the original assignment, see the assignment handout at
[cs336_spring2025_assignment4_data.pdf](./cs336_spring2025_assignment4_data.pdf)
Check out the [glossary of terms](./glossary.md) for this assignment.

Check out useful [lectures from CS336 at Stanford](https://github.com/stanford-cs336/spring2025-lectures).

If you see any issues with the assignment handout or code, please feel free to
raise a GitHub issue or open a pull request with a fix. Any improvements of the existing codebase
(including adaptations from Stanford to UHM workflows, modifications of PDF, etc) will be rewarded with extra points.

## Setup

This directory is organized as follows:

- [`./cs336-basics`](./cs336-basics): directory containing a module
  `cs336_basics` and its associated `pyproject.toml`. This module contains the staff 
  implementation of the language model from assignment 1. You will use this training code
  to train an LM on your filtered data. You should not modify the training logic, since
  your leaderboard submission must use it exactly.
- [`./cs336_data`](./cs336_data): This folder is basically empty! This is the
  module where you will implement code to filter and process the data.

Visually, it should look something like:

``` sh
.
├── cs336_basics  # A python module named cs336_basics
│   └── ... an optimized training implementation ...
├── cs336_data  # TODO(you): code that you'll write for assignment 4
│   ├── __init__.py
│   └── ... TODO(you): any other files or folders you need for assignment 4 ...
├── README.md
├── pyproject.toml
└── ... TODO(you): other files or folders you need for assignment 4 ...
```

As in previous assignments, we use `uv` to manage dependencies.

## Submitting

To submit, run `./test_and_make_submission.sh` . This script will install your
code's dependencies, run tests, and create a gzipped tarball with the output. We
should be able to unzip your submitted tarball and run
`./test_and_make_submission.sh` to verify your test results.


## ECE405 Assignment instructions

Follow along the [CS336@Stanford handout](./cs336_spring2025_assignment4_data.pdf) with small deviations:
1. What the code looks like: clone https://github.com/igormolybog/ece405-assignment2-data.git
2. How to submit: You will submit the report on the assignment to [Assignment Submission Form](https://docs.google.com/forms/d/e/1FAIpQLScJg_QkwjKux3xKeM-EOmZyvA6zlbVIrf_lxN_qoCFoxdqTrg/viewform?usp=sharing&ouid=111841773839267096112). The code does not have to be attached as long as you include links to the main GitHub branch where your code lives and links to all of the Colab notebooks if applicable.
    - You don't need to submit to leaderboard.
3. None of the data or tools are pre-downloaded or pre-installed for you. The handout describes the steps to get them (e.g. download specific WARC files or fastText library). You should follow the steps yourself.
4. You can use warcio library (ArchiveIterator) to iterate through WARC.
5. For Section 2.7, quality_classifier (a), you can also download the Wikipidea URLs from [here](https://drive.google.com/file/d/1hjlgyWSuMRDf-G0AXgGBWZmvjPJdDWc7/view?usp=sharing). To train a classifier, you can prepare a training file (e.g., "quality_train.txt") where each line is formatted as follows:
  ``` 
    __label__high This document is from Wikipedia and is high-quality.
    __label__low This document is from a random crawl and might be low-quality.
  ```
  - This file can be used as the input to the fasttext.train_supervised
6. Problems listed in Section 4 are not required and can be skipped. This is primarily due to the absence of the filtered WARC data. However, the problems can be implemented using data from any other WARC file collections and submitted for extra credit even after the deadline for Assignment 2 (and before the last day of the class) 
    - You may leverage CPUs located on Koa cluster to solve the problems listed in Section 4
