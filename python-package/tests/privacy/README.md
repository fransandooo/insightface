# Privacy test suite (Privacy and Big Data, KU Leuven, 2026-2027)

Privacy tests for the face-recognition office access system described in our
Privacy Impact Assessment. Each file covers one privacy issue from the report.
The tests are written to **fail** until the matching recommendation is
implemented. A failing test means the privacy issue is still open.

| File | Issue | What it checks |
|---|---|---|
| `test_p1_template_protection.py` | P1: plaintext, linkable templates | embeddings are not stored in clear text and a stolen BLOB cannot be matched with the public model |
| `test_p2_erasure.py` | P2: incomplete erasure | deleting an employee removes their samples, crop file and every trace in the SQLite file |
| `test_p3_access_log.py` | P3: function creep through the access log | access events expire after a retention period and never contain biometrics |
| `test_p4_bystanders.py` | P4: non-enrolled people | results for unmatched people carry no face or body template |

Tests whose names say the system still works (`..._still_identify_...`,
`..._still_recognised`) are utility guards. They must keep passing after a fix.

No model weights, GPU or network are needed. The storage tests use a temporary
SQLite file, and `PersonAnalysis` runs with fake models.

## Run

From the `python-package` folder, with the project virtual environment active:

```bash
pytest tests/privacy -v
```

Short summary, one line per failure:

```bash
pytest tests/privacy -q --tb=line -W ignore::DeprecationWarning
```
