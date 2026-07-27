# Automated BI pipeline with churn prediction and written weekly reporting

**[Live dashboard →](https://aykhaled-bi-automation-dashboard.streamlit.app/)**
· [Source →](https://github.com/aykhaled/bi-automation-dashboard)

---

## The problem

Most small and mid-sized retailers run reporting by hand. Someone exports a
transaction file, cleans it in a spreadsheet, rebuilds the same pivot tables,
and writes a summary email. It takes hours, it happens weekly, and the output
is backward-looking — it tells you what revenue *was*, never which customers
are about to stop buying.

I built the automated version of that process end to end, on a public dataset
of 1.07 million real retail transactions with all its genuine mess intact:
cancellations, missing customer IDs, service-charge lines, price outliers, and
duplicate rows.

## What it does

**Cleans and audits, visibly.** Five documented rules reduce 1,067,371 source
rows to 1,003,214 usable ones, isolating 17,914 cancellations rather than
discarding them — the return rate is itself a KPI. Every rule reports how many
rows it removed and why, published as a Data Quality tab in the dashboard. You
can see exactly what the pipeline did to your data.

**Predicts churn and sizes the exposure.** A gradient-boosted model ranks
customers by likelihood of not purchasing in the next 90 days, identifying
**427 customers who between them generated GBP 306,494 over the past year**.
That is a call list a marketing lead can work on Monday morning, downloadable
as a CSV.

**Writes the summary.** Every figure is computed in Python; a language model
turns those figures — and only those figures — into a short executive summary.
It cannot invent a number because it never sees the underlying data.

**Runs unattended.** A scheduled CI job regenerates the report weekly and fails
loudly if anything goes wrong. Full pipeline runtime: 22 seconds.

## What it found

The model surfaced a pattern the business would want to act on: churn in this
retailer is a **post-Christmas problem**. Across ten measurement points,
90-day churn ranges from 42.5% to 69.4%, peaking for customers active into the
December gift rush. That argues for a January reactivation campaign — a
specific, dated action rather than a general recommendation.

The pipeline also caught a reporting trap. One order of GBP 168,470 — 33% of
the week's gross revenue — was cancelled the same day. Reported naively, that
week shows 66% revenue growth. Actual growth was 11%. The report leads with net
revenue, names the invoice, and suppresses the metrics that order distorts.

## Why it transfers

The pipeline reads a single configuration file mapping your column names to its
internal vocabulary. **Pointing it at a different company's data is a
configuration change, not a rebuild.** Currency, churn window, cancellation
convention, and reporting audience are all settings.

Verified to produce bit-identical results on macOS and Windows — same inputs,
same numbers.

---

**Stack:** Python · pandas · scikit-learn · Streamlit · Plotly · OpenAI ·
GitHub Actions