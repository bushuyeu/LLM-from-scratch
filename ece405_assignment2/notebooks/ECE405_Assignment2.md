# ECE405 Assignment 2 — Written Responses

## Section 2: Filtering Common Crawl

### 2.1 Looking at the data (look_at_cc)

File: `/notebooks/ECE405_Assignment2.ipynb` - section 2.1

#### (a) First page in WARC file

The first response record in the WARC file has the URL `http://0371rykj.com/ipfhsb/34.html`, crawled on 2025-04-17. The page is no longer accessible. From the raw HTML, the `<title>` and `<meta>` tags contain HTML-encoded Chinese characters that decode to explicit adult content keywords, while the actual `<body>` content is about Shanghai Linpin Instrument Stock Co Ltd, a manufacturer of temperature and humidity testing chambers.

#### (b) Corresponding WET file

The WET extraction of this page begins with the decoded explicit Chinese keywords from the title/meta tags, immediately followed by navigation menu elements, product specifications, company boilerplate, and news headlines. 

A text extractor should have kept only the product description and specifications. 

Training a model on text like this risks teaching it to reproduce SEO spam patterns, boilerplates and UI elements as if that was natural text. 

However, the product specification table contains useful information (temperature ranges, model dimensions, component details) that could be valuable for a model that needs to answer questions about this equipment.

#### (c) What makes a good training example

This example could be useful for training a model intended to assist with industrial equipment specifications or product information. It would not be useful for training a general-purpose English language model, since it consists of Chinese text polluted with explicit SEO spam keywords and navigation boilerplate.

#### (d) Annotate 25 WET records

| # | URL | Language | Domain | Type | Notes |
|---|-----|----------|--------|------|-------|
| 1 | 0371rykj.com | Chinese | SEO spam domain | Product page with SEO spam | Explicit keywords in title/meta, actual content is industrial equipment |
| 2 | chinatikfans.com | Chinese | Fan forum | Discuz forum blog post | Fan site for Thai actor Tik Jesdaporn; personal blog entry from 2010 |
| 3 | 13.usnccm.org | English | Academic (.org) | Conference homepage | 13th US National Congress on Computational Mechanics (2015, San Diego).  |
| 4 | utchat888.com | Chinese (Traditional) | Adult chat platform | Livestream profile page | Adult video chat service |
| 5 | 176766.cn | Chinese | SEO spam domain | Product page with SEO spam | Explicit keywords in title, actual content about some instruments |
| 6 | 178mh.com | N/A | Broken site | 404 error | Only 17 chars: (template not found) |
| 7 | tgtg97.com | Chinese (Traditional) | Adult chat platform | Livestream profile page | Same platform template as Record 4 |
| 8 | 18sex.v340.info | Chinese (Traditional) | Adult chat platform | Directory listing | Same platform template as Record 4 |
| 9 | klimtoren.be | Dutch | Education blog (.be) | Teacher classroom blog | Short birthday post |
| 10 | mysch.gr | Greek | Education (.gr) | Forum search page | Greek education support helpdesk |
| 11 | mysch.gr | Greek | Education (.gr) | Forum login page | Same site as #10, login page |
| 12 | yhxzseo.com | Chinese | Gambling/SEO spam | Fake app landing page | Gambling platform disguised as tech review site |
| 13 | 20com20.fr | Turkish | Tech documentation (.fr) | Apache HTTP docs sitemap | Auto-translated Apache 2.4 documentation. |
| 14 | 24ktcasino.net | English | Gambling blog | Blog article | Article about Laos casinos. |
| 15 | 2kgames.eu | English | Broken site | 404 error | Only 34 chars: "404 Not Found" from nginx |
| 16 | yizhangting.com | Chinese | Gambling/SEO spam | Fake health article | Lottery platform content injected into health site template |
| 17 | 303323.com | Chinese (Traditional) | Medical devices | Product article | Article about electrocautery in minimally invasive GI surgery. |
| 18 | 30bad.com | Chinese | Pirate streaming | Anime streaming page | Streaming site for anime with boilerplate UI |
| 19 | 312001.net | Chinese | Healthcare (.net) | Community health center | Shaoxing community health center website. |
| 20 | mwe075.com | Chinese (Traditional) | Adult chat platform | Livestream profile page | Same adult chat template as Record 4 |
| 21 | schoollibrary.edu.pe.ca | English | School library (.edu) | Library catalog search | PEI school library OPAC search results. |
| 22 | haaxz.com | Chinese (Traditional) | Adult chat platform | Livestream profile page | Same adult chat template as Record 4 |
| 23 | haaxz.com | Chinese (Traditional) | Adult chat platform | Livestream profile page | Same domain and template as 4 |
| 24 | 387tel.com | Chinese (Traditional) | Adult chat platform | Video chat index | Same domain and template as 4 |
| 25 | 3diasdemarzo.blogspot.com | Spanish | Blogspot | Political blog | 2005 Spanish political blog about 11-M bombing investigation. |

**Number of examples until a "high-quality" page**: Arguably **25** — Record 25 (the Spanish political blog) is the first page with substantive, coherent, original written content. Record 3 (USNCCM conference) is structurally clean but mostly navigational. Record 14 (Laos casino blog) has some substance but is gambling-related. The majority of the first 25 records consist of adult chat platform pages (~8 of 25), SEO spam sites (~4), error pages (~2), navigation-heavy institutional pages, and other low-quality content. 

---

### 2.2 HTML to text conversion (extract_text)

File: `/notebooks/ECE405_Assignment2.ipynb` - section 2.2

#### (a) Text extraction function

File: `cs336_data/extract.py` — `extract_text_from_html_bytes(html_bytes)` decodes raw HTML bytes (detecting encoding if UTF-8 fails) and extracts visible text using Resiliparse's `extract_plain_text`. Adapter in `tests/adapters.py:run_extract_text_from_html_bytes`.

#### (b) Compare extraction methods

The Resiliparse extraction (10,165 chars) is nearly 3x longer than the WET extraction (3,496 chars) and significantly noisier — it retains HTML structural artifacts like `<th id="gckmo">`, whitespace, bullet point markers from hidden `display:none` divs, and nested navigation elements. 

The WET extraction is more compact and readable, stripping most structural markup and producing a flatter text representation, though it still includes navigation menus and sidebar content. For this particular page, the WET extraction appears better as training data: it is cleaner and more concise, whereas the Resiliparse output would inject HTML-like artifacts into a language model's training distribution.

---

### 2.3 Language identification (language_identification)

#### (a) Language identification function

File: `cs336_data/language_identification.py` — `identify_language(text)` returns `(language_code, confidence_score)` using fastText's `lid.176.bin` model. Adapter in `tests/adapters.py:run_identify_language`.

#### (b) Downstream issues from language filtering

Language ID errors can cause several downstream problems: 
- False negatives (e.g., English documents misclassified as another language) remove valuable training data; 
- False positives (e.g., non-English documents classified as English) can confuse the model and degrade generation quality. 
- Mixed-language documents(e.g., code with English comments on a Chinese site) are particularly problematic since the classifier must pick one label, potentially discarding useful content. 

In a higher-stakes deployment, these issues could be mitigated by using several language classifiers, applying language ID at a sub-document level.

#### (c) Manual language ID on 20 examples

File: `/notebooks/ECE405_Assignment2.ipynb` - section 2.3 (c)

Of 20 randomly sampled WET records, the classifier produced 19 correct predictions and 1 error:

- **Record 1** (`bagsguccistoer.com`): classified as Indonesian (`id`, 0.55) but the text is Thai. The low confidence reflects genuine uncertainty on a mixed-language spam page.


4 out of 20 documents are English (Records 4, 9, 10, 12, 19), though Record 4 is a nearly empty and Record 12 is a suspended-hosting notice. 

A confidence threshold of **0.80** would be suitable for English filtering: it retains all genuine English pages in this sample while excluding misclassified or ambiguous documents. Pages below this threshold (like Record 1 at 0.55) tend to be spam or mixed-language content that would be low-quality training data.

---

### 2.4 PII masking (mask_pii)

#### (a)–(c) PII masking functions

File: `cs336_data/pii.py` — `mask_emails(text)`, `mask_phone_numbers(text)`, `mask_ips(text)` use regex to replace PII with placeholder tokens (`|||EMAIL_ADDRESS|||`, `|||PHONE_NUMBER|||`, `|||IP_ADDRESS|||`). Each returns `(masked_text, count)`. Adapters in `tests/adapters.py`.

#### (4) Downstream problems from naive PII filtering

Naive regex-based PII filtering creates problems in both directions. False positives corrupt the training data: version numbers like "2.0.1.0" get masked as IP addresses, product codes or model numbers may match phone patterns, and email-like strings in code or URLs get unnecessarily redacted — all of which degrade data quality by replacing meaningful tokens with uninformative placeholders. False negatives are more dangerous: regex misses PII in non-standard formats (e.g., "john at gmail dot com", obfuscated emails), in non-Latin scripts, and entirely misses unstructured PII like names, physical addresses, or government ID numbers. A model trained on incompletely scrubbed data may memorize and reproduce real people's information, creating privacy and legal liability.

#### (5) False positives and negatives

File: `/notebooks/ECE405_Assignment2.ipynb` - section 2.4 (5)

Of 100 WARC pages examined, all 100 triggered at least one PII mask. Among 20 random examples, false positives dominate:

- **Phone regex** is the worst offender: it matches Blogger post IDs (Example 1: `blog-|||PHONE_NUMBER|||140222694`), calendar date sequences (Example 9), Chinese ICP registration numbers (Example 12: `蜀ICP备|||PHONE_NUMBER|||号`), government license numbers (Example 18), and article/element IDs (Examples 8, 19, 20). Roughly half of all phone "detections" are false positives matching arbitrary 10-digit numeric sequences.
- **Email regex** falsely matches git SSH URLs (Example 13: `git clone |||EMAIL_ADDRESS|||:packages/...`) and Blogger profile URLs (Example 1).
- **False negatives**: international phone formats like `+48 785 776 007` (Example 6, Polish numbers) are not caught because the regex only handles US 10-digit patterns. Pre-obfuscated emails like `[email protected]` (Example 17) are also missed.

The phone regex has the poorest precision — any 10-digit number matches regardless of context. Adding word boundaries, requiring separator characters, or restricting to known country formats would significantly reduce false positives.

---

### 2.5 Harmful content (harmful_content)

#### (a)–(b) Harmful content classifiers

File: `cs336_data/harmful_content.py` — `classify_nsfw(text)` and `classify_toxic_speech(text)` use Dolma/Jigsaw fastText models to classify text as `nsfw`/`non-nsfw` or `toxic`/`non-toxic` with confidence scores. Adapters in `tests/adapters.py`.

#### (3) Downstream problems from content filters

Aggressive content filtering might create biases in training data. Documents discussing sexual health, LGBTQ+ topics, etc. might be flagged as NSFW. 

Similarly, toxic speech classifiers trained on English data might penalize some texts that hold value. The resulting model inherits these biases: it may generate sterile, overly cautious responses on health topics.

On the other side, under-filtering might leave harmful content in the training data. A model trained on toxic text may reproduce slurs, hate speech, or harassment patterns, if toxic patterns are sufficiently common in the data.

#### (4) Classifier evaluation

File: `/notebooks/ECE405_Assignment2.ipynb` - section 2.5 (4)

Of 20 randomly sampled pages, all 20 were classified as `non-nsfw` and all 20 as `non-toxic`. Manual review found 3 NSFW false negatives:

- **False negatives (NSFW):** Record 2 (`50899.cn`) contains explicit Chinese pornographic keywords directly in the extracted text ("午夜亚洲影院在线观看", "黄网人妻视频", "午夜A级性爱") yet was classified as `non-nsfw (0.9999)`. Record 3 (`18sex.v340.info`) is a Taiwanese adult video chat platform — the domain itself contains "18sex" and the text includes adult service categories ("台妹區", "內地主播區", one-on-one credits) — yet classified `non-nsfw (1.0000)`. Record 6 (`adult.twlive520.info`) is the same platform template with "adult" in the domain — classified `non-nsfw (0.9876)`. All three are clear misses caused by the Dolma model being trained on English Jigsaw data, making it effectively blind to NSFW content in Chinese/Traditional Chinese.
- **Correct but low confidence:** Record 5 (Russian gambling spam on a hacked Kenyan site) scored the lowest NSFW confidence (0.9346). The `non-nsfw` label is correct — gambling content is not NSFW — but the lower confidence suggests the model is uncertain when encountering non-English text.
- **True negatives:** Records 1 (MIT ballroom dance), 4/9 (Cognitive Bias Foundation), 7 (Brazilian water tank manufacturer), 8 (French design blog), 10 (film production blog), 12 (French news), 13 (Spanish political blog), 15 (board game reviews), 17 (Belgian teacher blog), 18 (Danish window cleaning), 19 (Indian academic conference) — all correctly classified.
- **Note on content drift:** Record 13 (`3diasdemarzo.blogspot.com`) was a Spanish political blog in the WARC snapshot but the live URL now redirects to adult dating spam. Record 15 (`apidc.org`) showed board game reviews in the snapshot but the domain has since been hijacked. These illustrate how crawl snapshots age — domains get abandoned, expire, and are repurposed for spam, meaning re-crawled data may need reclassification.

The NSFW classifier's failure on Record 2 illustrates a key limitation: **monolingual classifiers miss harmful content in other languages**. This is especially problematic for Common Crawl, which is inherently multilingual. A production pipeline should either use multilingual NSFW models or apply language-specific classifiers after the language ID step.

The toxic classifier shows no variation (all 1.0000), which is expected since none of these pages contain English hate speech. Suggested thresholds: **0.40** for NSFW (to catch genuinely explicit content while tolerating borderline navigation pages), **0.50** for toxic as a starting point.

---

### 2.6 Quality Rules (gopher_quality_filters)

#### (b) Quality filter evaluation

*TODO: Run on extracted text, compare 20 filter predictions to own judgment.*

---

### 2.7 Quality Classifier (quality_classifier)

*(Implementation only — no written questions)*

---

## Section 3: Deduplication

*(Implementation only — no written questions)*

---

## Section 4: Leaderboard (Optional / Extra Credit)

### inspect_filtered_data

#### (a) 5 random examples from filtered data

*TODO*

#### (b) 5 random discarded examples

*TODO*

#### (c) Pipeline iterations

*TODO*
