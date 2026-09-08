# Benchmark Guide

Reproducible steps for comparing **ffs CLI**, **ffs MCP (warm server)**, and
**ripgrep (`rg`)** on a real repo.

---

## Quick start

```bash
# Build both tools in release mode first
cargo build --release -p ffs-cli --no-default-features
# or: cargo build --release -p ffs-cli  (with zlob, needs Zig)

# Pick a target repo to measure against (defaults to this repo)
REPO=.

python benches/run_throughput.py $REPO --queries 30    # cold spawn comparison
python benches/run_warm_mcp.py   $REPO --queries 30   # warm MCP server latency
```

All three scripts print per-query latency arrays suitable for `jq` or
`statistics` analysis.

---

## What each script measures

| Script | Mode | What it measures |
|--------|------|-----------------|
| `run_throughput.py` | **cold spawn** | Wall-clock for N sequential `ffs grep -l` spawns vs N sequential `rg -l` spawns. Each invocation is a fresh process — the real cost agents pay when calling Grep once per turn without MCP. |
| `run_throughput.py` | **warm MCP** | Same N queries routed through a single long-lived `ffs mcp` process (JSON-RPC over stdin/stdout). Measures the persistent-index advantage. |
| `run_warm_mcp.py` | **warm MCP detail** | Per-query p50/p95/p99 latency through MCP, with a configurable warmup phase. |
| `verify_correctness.py` | **correctness** | Ensures ffs and rg return the same file set for a battery of literal and regex queries. |

---

## When does ffs actually win vs plain `rg`?

Three factors determine where the crossover is:

1. **Number of files** — ffs's bigram prefilter skips non-candidate files in
   O(1); `rg` must walk-and-test every file. The advantage grows linearly with
   repo size and is negligible under ~200 files.

2. **Query selectivity** — the more selective the query (appears in <5% of
   files), the bigger ffs's skip rate. If the needle is in nearly every file
   (e.g. `import` in a small TypeScript project), rg's raw SIMD scan wins.

3. **Cold vs warm** — a fresh CLI invocation (cold) pays the same walk cost as
   `rg`, so gains are modest (~10-30%). The MCP server mode (warm) keeps the
   index in memory and avoids re-walking the tree entirely — this is where
   ffs's compounding advantage lives for agents making 20-50 queries per
   session.

### Rule of thumb

| Scenario | Winner |
|----------|--------|
| Repo < 200 files, few queries | `rg` (simpler, no index to build) |
| Repo > 500 files, selective queries, MCP mode | **ffs** (persistent index + bigram skip) |
| Agent session with 20+ queries | **ffs MCP** (amortised walk cost ~0) |
| Query matches >50% of files | `rg` (less index overhead, no prefilter gain) |

---

## Reproducing the README spawn benchmark

The tables in the README's `ffs vs rg` section were produced with:

```bash
# Requires: hyperfine, rg installed, ffs in target/release/
hyperfine --shell=none --warmup 3 --runs 30 \
  "./target/release/ffs grep poll --root $REPO -l" \
  "rg poll --root $REPO -l"
```

For a 3-round confirmation run (rounds = separate invocations of hyperfine,
all three must agree for ✅):

```bash
for i in 1 2 3; do
  echo "--- round $i ---"
  hyperfine --shell=none --warmup 3 --runs 30 \
    "./target/release/ffs grep poll --root $REPO -l" \
    "rg poll --root $REPO -l"
done
```

Only count "ffs wins" when ffs wins **all 3 rounds**.

---

## Building a reproducible benchmark from scratch

```bash
# 1. Clone a reference repo (~1k files is a good starting point)
git clone --depth=1 https://github.com/tokio-rs/tokio /tmp/tokio

# 2. Build release binaries
cargo build --release -p ffs-cli --no-default-features
cp target/release/ffs /usr/local/bin/ffs

# 3. Run all benchmarks
python benches/run_throughput.py /tmp/tokio --queries 30
python benches/run_warm_mcp.py   /tmp/tokio --queries 30 --warmup 5
python benches/verify_correctness.py /tmp/tokio

# 4. Compare with rg
hyperfine --shell=none --warmup 3 --runs 30 \
  "ffs grep fn /tmp/tokio -l" \
  "rg fn /tmp/tokio -l"
```

---

## Notes

- All benchmarks assume **release mode** (`cargo build --release`). Debug
  builds are 5-20x slower and produce misleading numbers.
- The `--no-default-features` build disables `zlob` (Zig globbing). To
  benchmark glob performance, use a build with `zlob` enabled (requires Zig
  installed).
- MCP latency is measured with raw JSON-RPC over stdin/stdout, not over a
  network socket — this is the mode Claude Code / Cursor actually use.
