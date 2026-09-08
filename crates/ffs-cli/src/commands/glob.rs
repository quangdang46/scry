use std::path::Path;

use anyhow::Result;
use clap::Parser;
use serde::Serialize;

use crate::cli::OutputFormat;
use crate::commands::pagination::footer;

#[derive(Debug, Parser)]
pub struct Args {
    /// Glob pattern (e.g. `**/*.rs`).
    pub pattern: String,

    /// Limit number of results emitted.
    #[arg(long, default_value_t = 200)]
    pub limit: usize,

    /// Skip this many matches before starting the page. Use together with
    /// `--limit` to page past a truncated result set (see the `[N-M of
    /// total]` footer).
    #[arg(long, default_value_t = 0)]
    pub offset: usize,
}

#[derive(Debug, Serialize)]
struct GlobResult {
    pattern: String,
    matches: Vec<String>,
    /// Matches skipped before this page (`--offset`).
    offset: usize,
    /// True when at least one more match exists beyond this page. Because
    /// the underlying scan stops at `offset + limit` matches (to stay fast
    /// on huge trees), `total_matches_at_least` is exact only when this is
    /// false — otherwise it's a lower bound.
    truncated: bool,
    /// Number of matches found up to the `offset + limit` cutoff. Exact
    /// total when `truncated` is false; a lower bound otherwise.
    total_matches_at_least: usize,
}

/// Delegate to the shared core glob function which handles Windows correctly
/// (falling back to `globset::Glob` + `ignore::WalkBuilder` on unsupported
/// platforms) and respects gitignore rules consistently.
fn glob_files(root: &Path, pattern: &str, limit: usize) -> Vec<String> {
    ffs_search::glob_matcher::glob_files(root, pattern, limit)
        .into_iter()
        .map(|rel| root.join(&rel).to_string_lossy().to_string())
        .collect()
}

pub fn run(args: Args, root: &Path, format: OutputFormat) -> Result<()> {
    let offset = args.offset;
    // Fetch offset+limit worth of matches so we can slice off `offset`
    // without asking the underlying scan for the whole (potentially huge)
    // result set.
    let take = offset.saturating_add(args.limit);
    let mut matches = glob_files(root, &args.pattern, take);
    let total_matches_at_least = matches.len();
    let truncated = matches.len() >= take && take > 0;
    if offset > 0 {
        matches = if offset >= matches.len() {
            Vec::new()
        } else {
            matches.split_off(offset)
        };
    }
    let returned = matches.len();
    let payload = GlobResult {
        pattern: args.pattern,
        matches,
        offset,
        truncated,
        total_matches_at_least,
    };
    super::emit(format, &payload, |p| {
        let mut out = String::new();
        for m in &p.matches {
            out.push_str(m);
            out.push('\n');
        }
        if !p.matches.is_empty() {
            let has_more = p.truncated || p.offset + returned < p.total_matches_at_least;
            out.push_str(&footer(
                p.total_matches_at_least,
                p.offset,
                returned,
                has_more,
            ));
        }
        out
    })
}
