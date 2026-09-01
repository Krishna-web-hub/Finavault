# FinVault sample test corpus

A coverage matrix, not a pile of documents. Every file exists to exercise a
specific code path; the "Exercises" column says which one, so a failure points
at a component rather than at "the demo broke".

All documents describe one consistent fictional world (Acme Capital Partners,
FY2026), so cross-document questions have verifiable right answers.

## Upload matrix

| File | Upload as | Format | Exercises |
| --- | --- | --- | --- |
| `firm_overview.txt` | **public** | txt | Baseline public tier — the only doc a `viewer` should ever see in full |
| `press_release_q3_results.md` | **public** | **md** | Markdown loader; public/internal split of the *same* Q3 figures |
| `quarterly_report_q1.txt` | **internal** | txt | Comparison series, point 1 of 3 |
| `quarterly_report_q2.txt` | **internal** | txt | Comparison series, point 2 of 3 (has a one-time $3.8M charge — tests non-comparability) |
| `quarterly_report.txt` (Q3) | **internal** | txt | Comparison series, point 3 of 3 |
| `board_minutes_2026_06.pdf` | **internal** | **pdf** | pypdf loader; hub document that references four others (graph edges) |
| `trading_desk_ledger.csv` | **internal** | **csv** | `tabular.summarize_csv` on numeric-heavy data; AnalystAgent's calculation tool |
| `client_accounts_extract.csv` | **confidential** | csv | Tabular profile on categorical data; links to account #48213 |
| `compliance_policy.txt` | **confidential** | txt | Confidential tier — `viewer` must be denied, `analyst` allowed |
| `incident_postmortem.txt` | **confidential** | txt | Named entities (person, date, incident) for graph extraction |
| `vendor_risk_assessment.docx` | **confidential** | **docx** | python-docx loader **+ prompt injection at a non-restricted tier** |
| `restricted_memo.txt` | **restricted** | txt | Restricted tier + injection (the original combined fixture) |
| `sar_filing_draft.txt` | **restricted** | txt | Restricted tier **without** injection — isolates ACL from guardrail |
| `unsupported_export.xlsx` | any | xlsx | **Negative:** expect HTTP 415 from `loaders.load_text` |
| `empty_upload.txt` | any | txt | **Negative:** zero-byte file through chunking |

Oversized upload (413) has no fixture on purpose — generate it at test time so a
20 MB blob never lands in git:

```bash
head -c 25000000 /dev/urandom | base64 > /tmp/oversized.txt
```

## Why the corpus is shaped this way

**Two injection fixtures, at two tiers.** With injection only in the restricted
memo, a passing test can't distinguish "the guardrail caught it" from "the ACL
denied it before the guardrail ran". `vendor_risk_assessment.docx` puts the same
attack behind confidential clearance, where an analyst *is* allowed to retrieve
the text — so the guardrail is the only thing standing between the attack and
the prompt.

**Two clean restricted docs.** `sar_filing_draft.txt` is restricted with no
adversarial content, so an ACL test proves clearance handling rather than
accidentally proving injection detection.

**Three quarters, not two.** Two points make a difference; three make a trend,
which is what a variance-based risk score in `ComparisonAgent` actually scores.
Q2's one-time charge exists so the heatmap has one metric that *shouldn't* be
compared naively.

**Same facts at two tiers.** The press release and the Q3 internal report state
the same revenue and EPS. A `viewer` asking "what was Q3 revenue?" should get an
answer from the public release; the internal-only detail (segment breakdown,
opex) should stay out of it. That is the externalization policy under test, and
it needs a public document that overlaps an internal one to be testable at all.

## Cross-document facts worth querying

These have exactly one correct answer, so a wrong answer is a real failure:

| Question | Expected | Spans |
| --- | --- | --- |
| Revenue growth from Q1 to Q3 FY2026? | $158.4M → $184.6M, +16.5% | 2 docs |
| Why did net income grow faster than revenue in Q3? | Expenses grew 11.7% vs revenue 21.2% | 1 doc |
| What caused incident INC-2026-0412? | Schema migration without an index | postmortem + PDF minutes |
| Which vendor's recovery objective breaches our tolerance? | Northwind — 8h RTO vs 4h tolerance | docx + PDF minutes |
| Total notional traded in the ledger? | ~$662.8M across 180 trades | csv |
| How many trades breached limits? | 7 | csv |
| Tell me about client account #48213 | **Restricted** — deny for viewer/analyst | memo + SAR + csv |

## Role expectations

| Role | public | internal | confidential | restricted |
| --- | --- | --- | --- | --- |
| `viewer` | yes | yes | **deny** | **deny** |
| `analyst` | yes | yes | yes | **deny** |
| `compliance_officer` | yes | yes | yes | yes* |
| `admin` | yes | yes | yes | yes* |

\* Retrievable, but still blocked from reaching the LLM by
`FINVAULT_ALLOWED_EXTERNAL_CLASSIFICATIONS` (default: `public,internal,confidential`).
A compliance officer asking about #48213 should get a *policy* refusal, not an
empty result — those are different code paths and worth telling apart.

## Org isolation

Org comes from the JWT, not the file. To test it: sign in as
`org_id=acme-capital`, upload anything, sign out, sign back in as
`org_id=northwind-partners`, and confirm the document is invisible — including
in the knowledge graph and the comparison document picker, not just in chat.

## Regenerating

`trading_desk_ledger.csv` (seeded, deterministic), `vendor_risk_assessment.docx`,
`board_minutes_2026_06.pdf`, and `unsupported_export.xlsx` are generated. The
`.txt` and `.md` files are hand-written and safe to edit directly — just keep the
figures consistent with the table above.
