# BI Automation Dashboard

An automated analytics pipeline that ingests raw transaction data, cleans it,
predicts which customers are about to churn, and writes an executive summary on
a schedule — replacing a recurring manual reporting cycle.

**[Live dashboard →](https://aykhaled-bi-automation-dashboard.streamlit.app/)**

---

## What it does

| Stage | Output |
|---|---|
| **Ingest** | Schema validation against a column mapping; fails loudly on mismatch |
| **Clean** | Five documented rules; emits a data quality report as a deliverable |
| **Features** | RFM plus behavioural features, built as time-separated snapshots |
| **Models** | Churn ranking (gradient boosting) and daily revenue anomaly detection |
| **Report** | Metrics computed in Python, narrative written by an LLM, rendered to HTML |

Run the whole chain or any single stage:

```bash
python -m src.pipeline --stage all
python -m src.pipeline --stage clean
python -m src.pipeline --stage all --config config/client_acme.yaml
```

Full pipeline: **~22 seconds on 1.07M rows**.

---

## Results on the reference dataset

| | |
|---|---|
| Source rows | 1,067,371 |
| Retained after cleaning | 1,003,214 (94.0%) |
| Returns isolated | 17,914 |
| Net revenue | GBP 18,926,266 (3.65% return rate) |
| Customers / products / orders | 5,852 / 4,878 / 39,516 |
| Churn model | AUC 0.737, 1.49× lift at the top decile |
| Customers flagged at risk | 427, carrying GBP 306,494 in trailing-12-month revenue |
| Revenue anomalies detected | 29 across 604 trading days |

Verified bit-identical on macOS ARM and Windows x86 — same inputs, same
figures, including the model metrics.

---

## Config-driven by design

Every module reads `config/config.yaml`. Onboarding a different company is a
configuration change, not a rebuild:

```yaml
columns:
  transaction_id: Invoice
  date: InvoiceDate
  customer_id: Customer ID
  product_id: StockCode
  quantity: Quantity
  unit_price: Price
  region: Country

business_rules:
  currency: GBP
  cancellation_prefix: "C"
  churn_window_days: 90
  product_code_pattern: "^\\d{5}"
```

`src/ingest.py` is the only module aware of client column names. It renames
everything to a canonical vocabulary once; every downstream module speaks only
that vocabulary. Nothing downstream can break when the source schema changes,
because nothing downstream knows where the data came from.

---

## Design decisions

**Time-separated snapshots, not a random split.** A *snapshot* is features
computed strictly before a cutoff date plus a label drawn from the window that
follows. Training data is a stack of ten such snapshots; scoring is one
unlabelled snapshot at the end of the data. Temporal leakage is structurally
impossible through this interface. Training snapshots are additionally spaced
so their label windows close before validation begins — stricter than "split by
time".

**Ranking, not calibrated probability.** Churn in this dataset is strongly
seasonal: across the ten training snapshots, 90-day churn ranges from 42.5% to
69.4%, peaking for windows that fall after the December gift season. Absolute
probabilities drift with the scoring date; rank order does not. Revenue at risk
is therefore reported as the *observed* trailing-12-month revenue of the
top-decile cohort, not as `sum(p × monetary)`. Validated with AUC and
precision-at-decile rather than calibration curves.

**Recency normalised to each customer's own rhythm.** `gap_ratio` divides days
since last order by that customer's mean inter-purchase gap. A weekly buyer
silent for 40 days is a very different signal from a quarterly buyer silent for
40.

**The pipeline writes; the dashboard only reads.** `app/dashboard.py` never
trains a model, never touches raw data, and never recomputes a business metric.
It renders artifacts from `data/processed/`. This keeps the deployed footprint
under 12MB, makes cold starts fast, and guarantees the dashboard and the weekly
report can never disagree — they read the same files.

**Observed trading days only.** This retailer does not trade on Saturdays.
Reindexing the daily series to a full calendar would inject artificial zeros,
every one of which would trip the anomaly detector.

---

## LLM narrative: grounding, and what grounding does not fix

The weekly summary is written by `gpt-4o-mini`, but every number is computed
deterministically in Python first. The model receives only the finished metrics
dict — never raw data — and performs no arithmetic.

Grounding eliminates fabricated *numbers*. It does not eliminate fabricated
*meaning*. Across four iterations the model, given entirely correct figures,
still invented causal explanations ("reflecting fewer but higher-value
transactions") and recharacterised an observed figure as a forecast ("potential
lost revenue"). Two mechanisms address that:

1. **Semantic context in the data.** `customer_risk.basis` carries the
   methodology note straight from `models.py`, so the caveat cannot drift from
   the method that produced the number. `order_concentration.affects` names the
   metrics a dominant order distorts.
2. **Withholding.** Fields the model must not report are removed from its input
   entirely (`for_narrative()`). The HTML table still renders them; only the
   model's view is narrowed.

The distinction that emerged:

| Constraint | Reliability | Enforcement |
|---|---|---|
| Never invent a number | High | Prompt |
| Never state a cause | High | Prompt |
| Use this exact framing | Moderate | Prompt + context in data |
| Omit or caveat these metrics | Low | Remove from input |

If the API key is absent or the call fails, the pipeline emits a deterministic
summary and marks it ungrounded rather than crashing. An unattended weekly job
must not break the build.

**Single-order concentration.** Gross revenue in the reference period is
dominated by one 80,995-unit order (invoice 581483, 33% of gross) that was
cancelled the same day. The report leads with net revenue, flags the invoice
explicitly, and suppresses the metrics that order distorts.

---

## Scheduling

`.github/workflows/weekly-report.yml` runs `--stage report` every Monday at
07:00 UTC, and on manual dispatch. It reads committed artifacts only, so it
needs no raw data. The run fails loudly if the narrative falls back to
deterministic mode, and posts the generated summary to the run page.

Requires an `OPENAI_API_KEY` repository secret.

---

## Running locally

```bash
uv python pin 3.11
uv venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
uv pip install -r requirements.txt
```

Download `online_retail_II.csv` to `data/raw/` (see Dataset below), then:

```bash
python -m src.pipeline --stage all
streamlit run app/dashboard.py
```

For the LLM narrative, put `OPENAI_API_KEY=sk-...` in a `.env` file at the repo
root. Without it the pipeline still runs and uses the deterministic fallback.

---

## Environment notes

**Version pins are deliberate — do not relax them.** `pyarrow 25.0.0` with
`numpy 2.4.6` segfaults on macOS ARM inside pandas' Arrow string conversion.
The crash surfaces at whatever Arrow conversion runs next, so it presents as an
application bug rather than a dependency one. `pyarrow<19` with `numpy<2.2` is
the stable pairing. The pins also guarantee the deploy target resolves the same
versions that were tested.

**Windows `.env` encoding.** PowerShell's `Out-File -Encoding utf8` writes a
BOM, which makes `python-dotenv` parse the first variable as
`\ufeffOPENAI_API_KEY`. Use
`[System.IO.File]::WriteAllText("$PWD\.env", "OPENAI_API_KEY=sk-...")` instead.

**Streamlit Cloud.** Set Python 3.11 in Advanced Settings at app creation.
Newer defaults break the build chain.

**Regenerating artifacts.** `data/processed/` is committed so the dashboard and
the CI report run without raw data. Paths in `quality_report.json` are
normalised with `as_posix()`, so artifacts are identical from any OS.

---

## Dataset

Chen, D. (2012). *Online Retail II* [Dataset]. UCI Machine Learning Repository.
<https://doi.org/10.24432/C5CG6D> — licensed CC BY 4.0.

1,067,371 transactions from a UK-based online gift retailer, December 2009 to
December 2011. Chosen because its data quality problems are real rather than
synthetic: cancellations as negative quantities, ~23% missing customer IDs,
non-product service lines, price outliers, and duplicate rows.

`data/raw/` is gitignored. `data/processed/` is committed.

---

## Stack

Python 3.11 · pandas · scikit-learn (`HistGradientBoostingClassifier`) ·
Streamlit · Plotly · Jinja2 · OpenAI · GitHub Actions · Streamlit Community Cloud