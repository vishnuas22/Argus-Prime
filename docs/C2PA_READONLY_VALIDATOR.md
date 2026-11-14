# C2PA Read-Only Validator (P9)

**Spec:** `p9-c2pa-readonly-v1`
**Schema:** `p9.c2pa_read.v1`
**Status:** standalone transparency surface; not wired into detection.

## What this is

A read-only C2PA / Content Credentials reader. It reports what a file's manifest
declares and whether the manifest's hard binding matches the file's bytes. It is
**transparency-only**: it is not imported by the court, registers no channel,
moves no pointer or threshold, and changes no verdict.
**C2PA is not a detection gate.**

The validator reports five manifest states:

- `absent` — no C2PA manifest was found. Absence is not evidence of either
  origin: most real and generated content carries no manifest.
- `well_formed_valid` — a resolvable active manifest parsed and validated, but
  it is not trusted. The in-repo fixture is this state.
- `trusted` — a resolvable active manifest validated with a clean status list,
  an exact `signingCredential.trusted` success code, and a non-empty
  `signature_info`. This state is strict by design.
- `invalid` — the manifest parsed but failed validation, or no active manifest
  is resolvable from a parsed store.
- `error` — the reader, the JSON payload, or the manifest object could not be
  processed.

## What it does not prove

- **Absence is not evidence.** A stripped manifest does not make an image real
  or fake.
- **Hard binding is not capture reality.** `hard_binding` proves that the bytes
  match the manifest's `dataHash`/`bmffHash`; it does not prove that the scene
  was captured by a camera or that the declared origin is true.
- **Manifests can be stripped or recaptured.** Re-encoding, screenshots, and
  recapture destroy or replace provenance. A missing or replaced manifest is a
  normal laundering outcome, not a detector signal.
- **Signed fakes exist.** A valid signature can attest a false or incomplete
  claim. Signing is an identity/claim mechanism, not a truth mechanism.
- **Trust is not world truth.** Even a `trusted` result means the configured
  trust anchors accepted the signer and the claim's status codes were clean; it
  does not authenticate the depicted event.
- **Revocation, timestamp, and trust-anchor limits apply.** The reader does not
  refresh revocation data, does not establish a trusted timestamp, and does not
  manage trust anchors. A validator's trust decision is only as current and as
  scoped as its trust store.

## The in-repo test chain

The fixture chain is a development/test credential, not a production trust
anchor. **The in-repo test chain is untrusted and must never be reported as trusted.**
The fixture is expected to be `well_formed_valid` with
`trusted=false` and `hard_binding=true`. A tampered copy is `invalid` with
`hard_binding=false`; a truncated file is `error`.

The `trusted` state has **no trusted fixture** in this repo. Its logic is
tested only with synthetic manifest dictionaries. Do not describe `trusted` as
end-to-end verified against a real trust store.

## Claims are declarations

`ai_claimed` and `camera_claimed` are text/assertion heuristics over the active
manifest. They report what the manifest declares, not what the pixels are.
`camera_claimed` recognizes capture-oriented markers such as `digitalCapture`,
`compositeCapture`, `c2pa.captured`, `stds.exif`, and `c2pa.capture`. It does
**not** treat `c2pa.hash.data` as a camera marker. Claims are declarations, not
proof.

## Hard binding

`hard_binding` is independent of `trusted`. It is true when the file validates
and the active manifest's success codes contain `assertion.dataHash.match` or
`assertion.bmffHash.match`, and no `dataHash.mismatch` or `bmffHash.mismatch`
code is present. A well-formed but untrusted manifest can still be hard-bound;
the in-repo fixture is exactly that case. BMFF hard binding is supported and is
synthetic-tested only.

## Read-only, offline, bounded

- The validator reads the **original file bytes** (no pixel decode) and passes
  them to `c2pa.Reader.try_create`.
- The read is **bounded** to `MAX_BYTES` (128 MiB by default, configurable with
  `--max-bytes`). The file is read with a `max_bytes + 1` ceiling.
- There is **no network** access: no HTTP client, no remote manifest fetch, no
  OCSP/CRL fetch.
- There is **no trust store** mutation: no settings, no context configuration,
  no trust-anchor installation. The validator never writes files.
- Reader exceptions other than a missing-manifest signal become `error`; the
  reader is always closed.

## CLI and API contract

Library:

```text
SCHEMA_VERSION = "p9.c2pa_read.v1"
MAX_BYTES = 134_217_728
MANIFEST_STATES = ("absent", "well_formed_valid", "trusted", "invalid", "error")

classify_manifest(manifest, *, path="<memory>", error=None) -> dict
read_c2pa(path, *, max_bytes=MAX_BYTES) -> dict
main(argv=None) -> int
```

CLI:

```text
./.venv/bin/python deepfake/scripts/c2pa_read.py PATH [--max-bytes N] [--pretty]
```

Stdout is exactly one JSON object plus a newline. Diagnostics go to stderr.
`--pretty` uses `indent=2`; the default is compact, sorted JSON.

The JSON object has exactly these 11 keys: `schema_version`, `path`,
`manifest_state`, `trusted`, `hard_binding`, `ai_claimed`, `camera_claimed`,
`signer_issuer`, `validation_state`, `validation_status_codes`, `error`.

Exit codes:

- `0` — analyzed (including `absent`, `well_formed_valid`, `trusted`,
  `invalid`, and reader/JSON error states).
- `2` — file missing, unreadable, too large, or invalid max-bytes; a JSON error
  object is still emitted.
- `1` — unexpected internal error; a JSON error object is emitted.
- argparse usage errors use the standard exit `2` with no JSON.

## Score and booking

This surface is a roadmap item estimated at **+0.4 system** and **0 detection**.
The court baseline remains **64.4/100**. The score is **unbooked** until a fresh
sentinel approves the receipt and the user's directive covers this no-wiring
item. No detection claim is made in any case.

## Residual risks

- The `trusted` path is logic-tested only; no trusted fixture exists.
- Claim heuristics can mislabel a manifest that names a generator or capture
  device without the corresponding origin.
- The 128 MiB bound still allocates a bounded buffer for large inputs.
- Native reader behavior and future c2pa/SDK drift can change status codes;
  tests pin c2pa-python 0.37.10 and fail rather than loosen.
- C2PA adoption and trust-store policy are external; this validator neither
  refreshes nor mutates them.
