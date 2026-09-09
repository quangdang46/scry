//! `ffs multi-grep` — multi-pattern OR search via Aho-Corasick.
//!
//! Wires the multi-needle path that already exists in the engine (`multi_grep_search`)
//! and MCP (`ffs_multi_grep`) into a first-class CLI subcommand. Patterns are
//! **literal** text (not regex); matching is OR across all needles in a single
//! pass per file.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Mutex;

use aho_corasick::AhoCorasickBuilder;
use anyhow::Result;
use clap::Parser;
use rayon::prelude::*;
use serde::Serialize;

use crate::cli::OutputFormat;

#[derive(Debug, Parser)]
#[command(after_help = "\
EXAMPLES:
  ffs multi-grep TODO FIXME HACK           # OR over 3 literals
  ffs multi-grep PrepareUpload prepare_upload
  ffs multi-grep -e TODO -e FIXME          # same, explicit -e flags
  ffs multi-grep TODO FIXME --limit 50
  ffs multi-grep TODO FIXME -l             # files-with-matches only
  ffs multi-grep TODO FIXME --format json

NOTES:
  Patterns are literal (not regex). For regex alternation use:
    ffs grep 'TODO|FIXME' --regex
  Engine/MCP multi-grep uses the same OR semantics (Aho-Corasick).")]
pub struct Args {
    /// Literal patterns to match (OR logic). At least one required unless `-e` is used.
    #[arg(required_unless_present = "pattern_flags")]
    pub patterns: Vec<String>,

    /// Additional pattern (repeatable). Same as positional patterns.
    #[arg(short = 'e', long = "pattern", value_name = "PATTERN")]
    pub pattern_flags: Vec<String>,

    /// Maximum matching lines emitted total across all files.
    #[arg(long, default_value_t = 200)]
    pub limit: usize,

    /// Match case sensitively (default: smart-case — insensitive only when
    /// every pattern is all-lowercase).
    #[arg(short = 's', long)]
    pub case_sensitive: bool,

    /// Stop after N matches per file. 0 = unlimited (default).
    #[arg(long = "max-count", default_value_t = 0)]
    pub max_count: usize,

    /// Output only the file paths (one per line) — like `rg -l`.
    #[arg(short = 'l', long = "files-with-matches")]
    pub files_with_matches: bool,
}

#[derive(Debug, Serialize)]
struct GrepHit {
    path: String,
    line: u32,
    text: String,
    /// Which pattern(s) matched this line (1-based pattern indices as given).
    #[serde(skip_serializing_if = "Vec::is_empty")]
    matched_patterns: Vec<String>,
    /// Byte ranges `[start, end)` of each match within `text`, for terminal
    /// highlighting. Omitted from JSON when empty.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    match_ranges: Vec<(u32, u32)>,
}

#[derive(Debug, Serialize)]
struct MultiGrepResult {
    patterns: Vec<String>,
    hits: Vec<GrepHit>,
    total_files_searched: usize,
    mode: &'static str,
    schema: &'static str,
}

fn collect_patterns(args: &Args) -> Result<Vec<String>> {
    let mut out = Vec::new();
    for p in args.patterns.iter().chain(args.pattern_flags.iter()) {
        let t = p.trim();
        if !t.is_empty() {
            out.push(t.to_string());
        }
    }
    if out.is_empty() {
        return Err(anyhow::anyhow!(
            "ffs multi-grep: need at least one non-empty pattern"
        ));
    }
    Ok(out)
}

/// Precomputed newline index for O(log n) byte-to-line mapping — avoids the
/// O(offset) linear rescan-from-zero that a naive `byte_to_line` would do for
/// every match (quadratic on hit-dense files). Built once per file.
struct NewlineIndex {
    positions: Vec<usize>,
}

impl NewlineIndex {
    fn build(haystack: &[u8]) -> Self {
        Self {
            positions: memchr::memchr_iter(b'\n', haystack).collect(),
        }
    }

    fn byte_to_line<'a>(&self, haystack: &'a [u8], offset: usize) -> (u32, usize, &'a [u8]) {
        let idx = match self.positions.binary_search(&offset) {
            Ok(i) => i + 1,
            Err(i) => i,
        };
        let line = (idx + 1) as u32;
        let line_start = if idx == 0 {
            0
        } else {
            self.positions[idx - 1] + 1
        };
        let line_end = haystack[line_start..]
            .iter()
            .position(|&b| b == b'\n')
            .map(|p| line_start + p)
            .unwrap_or(haystack.len());
        (line, line_start, &haystack[line_start..line_end])
    }
}

pub fn run(args: Args, root: &Path, format: OutputFormat) -> Result<()> {
    let patterns = collect_patterns(&args)?;

    let case_insensitive = if args.case_sensitive {
        false
    } else {
        // Smart case: insensitive only when ALL patterns are lowercase.
        !patterns.iter().any(|p| p.chars().any(|c| c.is_uppercase()))
    };

    let ac = AhoCorasickBuilder::new()
        .ascii_case_insensitive(case_insensitive)
        .build(&patterns)
        .map_err(|e| anyhow::anyhow!("aho-corasick build failed: {e}"))?;

    let files = super::walk_files(root);
    let total_files = files.len();
    let limit = args.limit;
    let max_count = if args.max_count == 0 {
        usize::MAX
    } else {
        args.max_count
    };

    let hits_mutex: Mutex<Vec<GrepHit>> = Mutex::new(Vec::new());
    let hit_counter = AtomicUsize::new(0);
    let stop = AtomicBool::new(false);

    files.par_iter().for_each(|path: &PathBuf| {
        if stop.load(Ordering::Relaxed) {
            return;
        }
        let Ok(content) = std::fs::read(path) else {
            return;
        };
        let probe = &content[..content.len().min(8 * 1024)];
        if probe.contains(&0u8) {
            return;
        }

        // Collect matches keyed by line so we de-dupe and attach which patterns hit.
        // pattern_id from Aho-Corasick is the index into `patterns`.
        // (text, matched_patterns, match_ranges)
        type LineHits = (String, Vec<String>, Vec<(u32, u32)>);
        let mut by_line: std::collections::BTreeMap<u32, LineHits> =
            std::collections::BTreeMap::new();

        let newline_index = NewlineIndex::build(&content);
        for (per_file, mat) in ac.find_iter(&content).enumerate() {
            if per_file >= max_count {
                break;
            }
            let (line, line_start, slice) = newline_index.byte_to_line(&content, mat.start());
            let pat = patterns[mat.pattern().as_usize()].clone();
            let entry = by_line.entry(line).or_insert_with(|| {
                (
                    String::from_utf8_lossy(slice).into_owned(),
                    Vec::new(),
                    Vec::new(),
                )
            });
            if !entry.1.contains(&pat) {
                entry.1.push(pat);
            }
            // Byte range of this match relative to the displayed line.
            let s = mat.start().saturating_sub(line_start);
            let e = mat.end().saturating_sub(line_start);
            let len = entry.0.len();
            if e > s {
                entry.2.push((s.min(len) as u32, e.min(len) as u32));
            }
            if args.files_with_matches && !by_line.is_empty() {
                break;
            }
        }

        if by_line.is_empty() {
            return;
        }

        let local_hits: Vec<GrepHit> = by_line
            .into_iter()
            .map(|(line, (text, matched_patterns, match_ranges))| GrepHit {
                path: path.to_string_lossy().into_owned(),
                line,
                text,
                matched_patterns,
                match_ranges,
            })
            .collect();

        let prior = hit_counter.fetch_add(local_hits.len(), Ordering::Relaxed);
        if prior >= limit {
            stop.store(true, Ordering::Relaxed);
            return;
        }

        if let Ok(mut guard) = hits_mutex.lock() {
            for h in local_hits {
                if guard.len() >= limit {
                    stop.store(true, Ordering::Relaxed);
                    break;
                }
                guard.push(h);
            }
        }
    });

    let mut hits = hits_mutex.into_inner().unwrap_or_default();
    hits.sort_by(|a, b| a.path.cmp(&b.path).then(a.line.cmp(&b.line)));
    hits.truncate(limit);

    if args.files_with_matches {
        let mut paths: Vec<String> = hits.iter().map(|h| h.path.clone()).collect();
        paths.sort();
        paths.dedup();
        hits = paths
            .into_iter()
            .map(|p| GrepHit {
                path: p,
                line: 0,
                text: String::new(),
                matched_patterns: Vec::new(),
                match_ranges: Vec::new(),
            })
            .collect();
    }

    let payload = MultiGrepResult {
        patterns,
        hits,
        total_files_searched: total_files,
        mode: "multi-literal-or",
        schema: "v1",
    };

    super::emit(format, &payload, |p| {
        let mut out = String::new();
        let path_spec = super::render::path_spec();
        let line_spec = super::render::line_spec();
        for h in &p.hits {
            if h.line == 0 {
                out.push_str(&super::render::colorize(&h.path, &path_spec));
                out.push('\n');
            } else {
                out.push_str(&super::render::colorize(&h.path, &path_spec));
                out.push(':');
                out.push_str(&super::render::colorize(&h.line.to_string(), &line_spec));
                if h.matched_patterns.is_empty() {
                    out.push_str(": ");
                    out.push_str(&super::render::colorize_matches(&h.text, &h.match_ranges));
                } else {
                    out.push_str(&format!(": [{}] ", h.matched_patterns.join("|")));
                    out.push_str(&super::render::colorize_matches(&h.text, &h.match_ranges));
                }
                out.push('\n');
            }
        }
        if p.hits.is_empty() {
            out.push_str(&format!(
                "[no matches across {} files for patterns {:?}]\n",
                p.total_files_searched, p.patterns
            ));
        }
        out
    })
}
