use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

use anyhow::Result;
use clap::Parser;
use memchr::memmem;
use serde::Serialize;

use crate::cli::OutputFormat;
use crate::commands::pagination::footer;

/// Read a file's bytes for searching: single open, single read_to_end (one
/// syscall, no second `File::open`, no per-file metadata syscall).
fn read_for_search(path: &Path) -> std::io::Result<Vec<u8>> {
    use std::io::Read;
    let mut file = std::fs::File::open(path)?;
    let mut buf = Vec::new();
    file.read_to_end(&mut buf)?;
    Ok(buf)
}

/// Push a path-only hit (-l / --files-without-match) with the shared
/// take/stop accounting. Extracted so the -l fast path and the inverted
/// variant share one accounting implementation.
fn push_path_hit(
    path: &Path,
    hits_mutex: &Mutex<Vec<GrepHit>>,
    hit_counter: &AtomicUsize,
    stop: &std::sync::atomic::AtomicBool,
    take: usize,
) {
    let prior = hit_counter.fetch_add(1, Ordering::Relaxed);
    if prior >= take {
        stop.store(true, Ordering::Relaxed);
        return;
    }
    if let Ok(mut guard) = hits_mutex.lock() {
        if guard.len() >= take {
            stop.store(true, Ordering::Relaxed);
            return;
        }
        guard.push(GrepHit {
            path: path.to_string_lossy().into_owned(),
            line: 0,
            text: String::new(),
            match_ranges: Vec::new(),
            context_before: Vec::new(),
            context_after: Vec::new(),
        });
    }
}

/// Streaming literal match: searches a file in chunks via BufReader, returning
/// true at the first match without allocating the full file content. Carries
/// a small tail overlap between chunks so a match that spans a chunk boundary
/// is never missed.
#[allow(dead_code)] // exercised by tests; superseded in the hot path by unified read_for_search
fn stream_match_literal(path: &Path, needle: &[u8], case_insensitive: bool) -> bool {
    use std::io::{BufReader, Read};
    let file = match std::fs::File::open(path) {
        Ok(f) => f,
        Err(_) => return false,
    };
    const CHUNK: usize = 64 * 1024;
    let overlap = needle.len().saturating_sub(1);
    let buf_size = CHUNK + overlap;
    let mut reader = BufReader::with_capacity(CHUNK, file);
    let mut buf = vec![0u8; buf_size];
    let mut filled = 0usize; // bytes currently in buf

    loop {
        // Shift any carried-over tail to the start.
        if filled > 0 && overlap > 0 {
            let tail_start = filled.saturating_sub(overlap);
            buf.copy_within(tail_start..filled, 0);
            filled -= tail_start;
        } else {
            filled = 0;
        }

        // Fill the rest of the buffer from the reader.
        let read_start = filled;
        let read_end = buf_size;
        let n = match reader.read(&mut buf[read_start..read_end]) {
            Ok(0) => {
                // Last chunk — search what we have without overlap padding.
                if filled > 0 {
                    return find_in_chunk(&buf[..filled], needle, case_insensitive);
                }
                break;
            }
            Ok(n) => n,
            Err(_) => break,
        };
        filled = read_start + n;

        if find_in_chunk(&buf[..filled], needle, case_insensitive) {
            return true;
        }
    }
    false
}

#[inline(always)]
#[allow(dead_code)] // used by stream_match_literal, currently test-only
fn find_in_chunk(haystack: &[u8], needle: &[u8], case_insensitive: bool) -> bool {
    if case_insensitive {
        // Case-insensitive: scan for first byte candidates, verify full needle.
        if needle.is_empty() {
            return true;
        }
        let first_lo = needle[0].to_ascii_lowercase();
        let first_hi = needle[0].to_ascii_uppercase();
        let rest = &needle[1..];
        for pos in memchr::memchr2_iter(first_lo, first_hi, haystack) {
            if pos + 1 + rest.len() <= haystack.len()
                && haystack[pos + 1..pos + 1 + rest.len()].eq_ignore_ascii_case(rest)
            {
                return true;
            }
        }
        false
    } else {
        memmem::find(haystack, needle).is_some()
    }
}

#[derive(Debug, Parser)]
#[command(after_help = "\
EXAMPLES:
  ffs grep TODO                            # smart-case literal search
  ffs grep '\\bTODO\\b' --regex            # forced regex (auto-detect would also pick this up)
  ffs grep -F '.is_file()'                 # force literal — '.' won't be a regex wildcard
  ffs grep --regex 'fn\\s+\\w+\\(' --root crates/  # signature-style regex over a sub-tree
  ffs grep -w error                        # whole-word match only
  ffs grep -l fixme                        # files-with-matches mode (one path per line)
  ffs grep -C 2 handleSubmit                # 2 lines of context before/after each match
  ffs grep TODO --offset 200                # page past the first 200 matches")]
pub struct Args {
    /// Pattern. Auto-detected as a regular expression when it contains any
    /// regex metacharacter (`.`, `*`, `+`, `?`, `^`, `$`, `[`, `(`, `|`, `\`).
    /// Force literal interpretation with `--fixed-strings`, or force regex
    /// with `--regex`.
    pub needle: String,

    /// Maximum lines emitted total across all files.
    #[arg(long, default_value_t = 200)]
    pub limit: usize,

    /// Skip this many matches before starting the page. Use together with
    /// `--limit` to page past a truncated result set (see the `[N-M of
    /// total]` footer).
    #[arg(long, default_value_t = 0)]
    pub offset: usize,

    /// Match case sensitively (default: false / smart-case when unset).
    #[arg(short = 's', long)]
    pub case_sensitive: bool,

    /// Match case insensitively, overriding smart-case (like `rg -i`).
    #[arg(short = 'i', long = "ignore-case", conflicts_with = "case_sensitive")]
    pub ignore_case: bool,

    /// Force regex interpretation (overrides auto-detection).
    #[arg(short = 'r', long)]
    pub regex: bool,

    /// Force literal / fixed-string interpretation (overrides auto-detection).
    #[arg(short = 'F', long = "fixed-strings", conflicts_with = "regex")]
    pub fixed_strings: bool,

    /// Require whole-word matches (wraps the pattern with `\b…\b`).
    #[arg(short = 'w', long = "word-regexp")]
    pub word_regexp: bool,

    /// Invert matching: select non-matching lines (like `rg -v`).
    #[arg(short = 'v', long = "invert-match")]
    pub invert_match: bool,

    /// Stop after N matches per file. 0 = unlimited (default).
    #[arg(long = "max-count", default_value_t = 0)]
    pub max_count: usize,

    /// Shorthand for `--max-count 1` per file (like `rg -m 1`).
    #[arg(short = 'm', long = "max-count-per-file")]
    pub max_count_per_file: Option<usize>,

    /// Output only the file paths (one per line) — like `rg -l`.
    #[arg(
        short = 'l',
        long = "files-with-matches",
        conflicts_with = "files_without_match"
    )]
    pub files_with_matches: bool,

    /// Output only files with NO match (like `rg --files-without-match`).
    #[arg(long = "files-without-match")]
    pub files_without_match: bool,

    /// Search hidden files and directories (like `rg --hidden`).
    #[arg(long)]
    pub hidden: bool,

    /// Don't respect `.gitignore` / `.ignore` files (like `rg --no-ignore`).
    #[arg(long)]
    pub no_ignore: bool,

    /// Search binary files as if they were text (like `rg -a/--text`).
    #[arg(short = 'a', long = "text")]
    pub text: bool,

    /// Include files matching the given glob (repeatable, like `rg -g`).
    /// Example: `-g '*.rs' -g '!target/**'`.
    #[arg(short = 'g', long = "glob")]
    pub globs: Vec<String>,

    /// Only show the matching part of each line (like `rg -o/--only-matching`).
    #[arg(short = 'o', long = "only-matching")]
    pub only_matching: bool,

    /// Group matches by file and enclosing symbol (like agentgrep).
    #[arg(long)]
    pub group: bool,

    /// Lines of context to show after each match (like `rg -A`).
    #[arg(short = 'A', long = "after-context", default_value_t = 0)]
    pub after_context: usize,

    /// Lines of context to show before each match (like `rg -B`).
    #[arg(short = 'B', long = "before-context", default_value_t = 0)]
    pub before_context: usize,

    /// Lines of context to show before AND after each match (like `rg -C`).
    /// Overridden per-side by `--after-context`/`--before-context` if those
    /// are also given.
    #[arg(short = 'C', long = "context", default_value_t = 0)]
    pub context: usize,
}

impl Args {
    fn context_lines(&self) -> (usize, usize) {
        let before = if self.before_context > 0 {
            self.before_context
        } else {
            self.context
        };
        let after = if self.after_context > 0 {
            self.after_context
        } else {
            self.context
        };
        (before, after)
    }
}

/// One emitted display row before it's wrapped into a `GrepHit`:
/// `(line number, display text, match ranges within that text)`.
type LineHit = (u32, String, Vec<(u32, u32)>);

#[derive(Debug, Serialize)]
struct GrepHit {
    path: String,
    line: u32,
    text: String,
    /// Byte ranges `[start, end)` of each match within `text`, for terminal
    /// highlighting. Omitted from JSON when empty (e.g. `-l` mode).
    #[serde(skip_serializing_if = "Vec::is_empty")]
    match_ranges: Vec<(u32, u32)>,
    /// Lines immediately before `line`, in file order. Only populated when
    /// `-B`/`-C` was requested.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    context_before: Vec<String>,
    /// Lines immediately after `line`, in file order. Only populated when
    /// `-A`/`-C` was requested.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    context_after: Vec<String>,
}

#[derive(Debug, Serialize)]
struct GrepResult {
    needle: String,
    hits: Vec<GrepHit>,
    total_files_searched: usize,
    /// "literal" or "regex" — whichever matcher actually ran.
    mode: &'static str,
    /// Matches skipped before this page (`--offset`).
    offset: usize,
    /// True when at least one more match exists beyond this page. Because
    /// the walk stops scanning as soon as `offset + limit` matches are
    /// found (to keep large repos fast), `total_matches_at_least` is exact
    /// only when `truncated` is false — otherwise it's a lower bound.
    truncated: bool,
    /// Number of matches found up to the `offset + limit` cutoff. Exact
    /// total when `truncated` is false; a lower bound otherwise.
    total_matches_at_least: usize,
    schema: &'static str,
}

/// Include/exclude glob filter from `-g` patterns (rg semantics):
/// `!`-prefixed patterns are excludes, the rest are includes. A file passes
/// when it matches no exclude AND (there are no includes OR it matches an
/// include). Matching is against the path relative to the search root, with
/// a basename fallback so `-g '*.rs'` matches at any depth.
struct GlobFilter {
    includes: globset::GlobSet,
    excludes: globset::GlobSet,
    has_includes: bool,
    has_excludes: bool,
}

fn build_glob_filter(patterns: &[String]) -> Result<Option<GlobFilter>> {
    if patterns.is_empty() {
        return Ok(None);
    }
    let mut inc = globset::GlobSetBuilder::new();
    let mut exc = globset::GlobSetBuilder::new();
    let mut has_includes = false;
    let mut has_excludes = false;
    for p in patterns {
        let (neg, pat) = match p.strip_prefix('!') {
            Some(rest) => (true, rest),
            None => (false, p.as_str()),
        };
        let mut gb = globset::GlobBuilder::new(pat);
        gb.literal_separator(true);
        let g = gb
            .build()
            .map_err(|e| anyhow::anyhow!("invalid glob {p:?}: {e}"))?;
        if neg {
            exc.add(g);
            has_excludes = true;
        } else {
            inc.add(g);
            has_includes = true;
        }
    }
    Ok(Some(GlobFilter {
        includes: inc
            .build()
            .map_err(|e| anyhow::anyhow!("glob set build failed: {e}"))?,
        excludes: exc
            .build()
            .map_err(|e| anyhow::anyhow!("glob set build failed: {e}"))?,
        has_includes,
        has_excludes,
    }))
}

impl GlobFilter {
    fn matches(&self, root: &Path, path: &Path) -> bool {
        let rel = path.strip_prefix(root).unwrap_or(path);
        let file_name = path.file_name().map(Path::new);
        let hit = |set: &globset::GlobSet| {
            set.is_match(rel) || file_name.is_some_and(|f| set.is_match(f)) || set.is_match(path)
        };
        if self.has_excludes && hit(&self.excludes) {
            return false;
        }
        if self.has_includes {
            return hit(&self.includes);
        }
        true
    }
}

/// Auto-detect: looks like a regex if it contains any of `.+*?^$[(|\` characters.
/// Mirrors the heuristic in `ffs::grep::has_regex_metacharacters` so CLI and MCP
/// agree on what a "literal" query looks like.
fn looks_like_regex(s: &str) -> bool {
    s.chars().any(|c| {
        matches!(
            c,
            '.' | '+' | '*' | '?' | '^' | '$' | '[' | '(' | '|' | '\\'
        )
    })
}

#[derive(Clone)]
enum Matcher {
    Literal {
        needle: Vec<u8>,
        case_insensitive: bool,
    },
    Regex(regex::bytes::Regex),
}

impl Matcher {
    fn build(args: &Args) -> Result<(Self, &'static str)> {
        // Smart case: -s forces sensitive, -i forces insensitive; otherwise
        // a pattern with any uppercase => sensitive (rg --smart-case).
        let smart_case_sensitive = if args.ignore_case {
            false
        } else {
            args.case_sensitive || args.needle.chars().any(|c| c.is_uppercase())
        };

        let use_regex = args.regex || (!args.fixed_strings && looks_like_regex(&args.needle));

        if use_regex {
            let mut pattern = args.needle.clone();
            if args.word_regexp {
                pattern = format!(r"\b(?:{})\b", pattern);
            }
            let re = regex::bytes::RegexBuilder::new(&pattern)
                .case_insensitive(!smart_case_sensitive)
                .multi_line(true)
                .build()
                .map_err(|e| anyhow::anyhow!("invalid regex {:?}: {e}", args.needle))?;
            Ok((Matcher::Regex(re), "regex"))
        } else {
            let needle_bytes = if smart_case_sensitive {
                args.needle.as_bytes().to_vec()
            } else {
                args.needle.to_lowercase().into_bytes()
            };
            Ok((
                Matcher::Literal {
                    needle: needle_bytes,
                    case_insensitive: !smart_case_sensitive,
                },
                "literal",
            ))
        }
    }

    /// Returns a lazy iterator of `(start, end)` byte offsets for each match.
    /// No intermediate allocation: matches are produced on demand, so `-l`
    /// mode can stop after the first match without scanning the whole file.
    fn find_iter<'a>(
        &'a self,
        haystack: &'a [u8],
    ) -> Box<dyn Iterator<Item = (usize, usize)> + 'a> {
        match self {
            Matcher::Literal {
                needle,
                case_insensitive,
            } => {
                let nlen = needle.len();
                if *case_insensitive {
                    // Case-insensitive ASCII: scan for (first, last) needle
                    // bytes via memchr2, then verify the interior with a fast
                    // case-folded compare. No lowercased copy, fully lazy.
                    let needle = needle.clone();
                    Box::new(CaseInsensitiveLiteralIter {
                        haystack,
                        needle,
                        pos: 0,
                    })
                } else {
                    // Case-sensitive literal: stream via memmem lazily (no
                    // intermediate position Vec).
                    Box::new(memmem::find_iter(haystack, needle).map(move |p| (p, p + nlen)))
                }
            }
            Matcher::Regex(re) => Box::new(re.find_iter(haystack).map(|m| (m.start(), m.end()))),
        }
    }

    /// Returns true if `haystack` contains at least one match. Used by
    /// files-with-matches mode: short-circuits at the first hit.
    fn is_match(&self, haystack: &[u8]) -> bool {
        match self {
            Matcher::Literal {
                needle,
                case_insensitive,
            } => {
                if *case_insensitive {
                    CaseInsensitiveLiteralIter {
                        haystack,
                        needle: needle.clone(),
                        pos: 0,
                    }
                    .next()
                    .is_some()
                } else {
                    memmem::find(haystack, needle).is_some()
                }
            }
            Matcher::Regex(re) => re.is_match(haystack),
        }
    }
}

/// Lazy case-insensitive ASCII literal iterator.
///
/// Finds candidate positions with `memchr2` on the needle's first byte
/// (both cases), then verifies the full needle with a case-folded compare
/// that relies on ASCII differing only in bit 0x20. Zero allocation.
struct CaseInsensitiveLiteralIter<'a> {
    haystack: &'a [u8],
    needle: Vec<u8>,
    pos: usize,
}

impl Iterator for CaseInsensitiveLiteralIter<'_> {
    type Item = (usize, usize);

    fn next(&mut self) -> Option<(usize, usize)> {
        if self.needle.is_empty() || self.haystack.len() < self.needle.len() {
            return None;
        }
        let first_lo = self.needle[0];
        let first_hi = first_lo.to_ascii_uppercase();
        let tail = &self.needle[1..];
        let max_pos = self.haystack.len() - self.needle.len();
        for pos in memchr::memchr2_iter(first_lo, first_hi, &self.haystack[self.pos..]) {
            let abs = self.pos + pos;
            if abs > max_pos {
                return None;
            }
            let candidate = &self.haystack[abs + 1..abs + self.needle.len()];
            if ascii_case_eq(candidate, tail) {
                self.pos = abs + 1;
                return Some((abs, abs + self.needle.len()));
            }
        }
        self.pos = self.haystack.len();
        None
    }
}

/// Fast ASCII case-insensitive byte-slice comparison (differ only in bit
/// 0x20). Both slices must be equal length.
fn ascii_case_eq(a: &[u8], b: &[u8]) -> bool {
    let len = a.len();
    let mut i = 0;
    while i + 8 <= len {
        let va = u64::from_ne_bytes(a[i..i + 8].try_into().expect("8-byte slice"));
        let vb = u64::from_ne_bytes(b[i..i + 8].try_into().expect("8-byte slice"));
        if va != vb {
            const MASK: u64 = 0x2020_2020_2020_2020;
            if (va | MASK) != (vb | MASK) {
                return false;
            }
        }
        i += 8;
    }
    while i < len {
        let ha = a[i];
        let hb = b[i];
        if ha != hb && (ha | 0x20) != (hb | 0x20) {
            return false;
        }
        i += 1;
    }
    true
}

/// Precomputed newline index for O(log n) byte-to-line mapping.
///
/// Builds once per file; each `byte_to_line` call becomes binary search
/// over sorted newline positions — critical for hit-dense patterns.
struct NewlineIndex {
    /// Byte offsets of each `\n` in the haystack, sorted ascending.
    positions: Vec<usize>,
}

impl NewlineIndex {
    fn build(haystack: &[u8]) -> Self {
        let positions: Vec<usize> = memchr::memchr_iter(b'\n', haystack).collect();
        Self { positions }
    }

    /// Map a byte offset to `(1-based line number, byte offset of line start, line slice)`.
    fn byte_to_line<'a>(&self, haystack: &'a [u8], offset: usize) -> (u32, usize, &'a [u8]) {
        // Binary search: find the last newline position <= offset.
        // The number of newlines before `offset` gives us the 0-based line index.
        let idx = match self.positions.binary_search(&offset) {
            Ok(i) => i + 1, // offset lands exactly on a newline -> next line
            Err(i) => i,    // i is the insertion point = count of newlines before offset
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
    if args.needle.is_empty() {
        return Err(anyhow::anyhow!(
            "ffs grep: needle is empty; pass a non-empty pattern"
        ));
    }
    let (matcher, mode) = Matcher::build(&args)?;

    // Bigram prefilter: only safe (and helpful) for literal patterns.
    // We try to load the persisted index; on miss we just scan everything.
    // `needle_bytes` is the case-folded literal we search for.
    let needle_bytes: &[u8] = match &matcher {
        Matcher::Literal { needle, .. } => needle.as_slice(),
        _ => &[],
    };
    let bigram = match &matcher {
        Matcher::Literal { needle, .. } if needle.len() >= 2 => {
            crate::cache::CacheDir::at(root).load_bigram_index(root)
        }
        _ => None,
    };

    // Bigram prefilter → a set of candidate paths; the search still walks the
    // whole tree ONCE (rg-style fused walk+search) and skips non-candidates via
    // O(1) membership. No prefilter → search every file in the walk. This keeps
    // a single parallel walk regardless of how scattered the candidates are.
    //
    // Correctness: a file may be skipped ONLY when it is both (a) not in the
    // candidate set AND (b) provably current (its (mtime, size) still matches
    // what the index recorded). Files added or edited since indexing are
    // force-scanned even if not candidates, so the prefilter never false-
    // negatives. We therefore carry (candidates, index) — the index only to
    // answer "is this file current?".
    type BigramOrdinals<'a> = std::collections::HashMap<&'a Path, usize>;
    let (candidate_paths, bigram_idx, bigram_ordinals): (
        Option<std::collections::HashSet<PathBuf>>,
        Option<&crate::bigram::GrepBigram>,
        Option<BigramOrdinals>,
    ) = match &bigram {
        Some(idx) => (
            idx.filter(needle_bytes)
                .map(|paths| paths.into_iter().map(PathBuf::from).collect()),
            Some(idx),
            Some(idx.ordinal_map()),
        ),
        None => (None, None, None),
    };
    // NOTE: we deliberately do NOT short-circuit when the candidate set is
    // empty. Files added since indexing are force-scanned by the walk (they
    // aren't in the index, so `is_current` is false), and such a file could
    // contain the needle even though no indexed file's bigrams matched. A
    // short-circuit here would be a false negative.
    // `total_files` (the bug-18 denominator) is the whole workspace. From the
    // bigram cache when present (no walk needed). When there's no cache we
    // count files during the fused search walk via `file_counter`, so we never
    // walk the tree twice.
    let total_files: usize = bigram.as_ref().map_or(usize::MAX, |idx| idx.file_count());
    let limit = args.limit;
    let offset = args.offset;
    // The walk stops as soon as `take` matches are found rather than `limit`,
    // so a page past the first one (`--offset`) doesn't force a full re-scan
    // of everything before it — we just scan a little further and slice.
    let take = offset.saturating_add(limit);
    let (before_context, after_context) = args.context_lines();
    // -m N is shorthand for --max-count N (rg semantics); explicit
    // --max-count wins when both are given.
    let max_count = match (args.max_count, args.max_count_per_file) {
        (0, None) => usize::MAX,
        (0, Some(m)) => m,
        (c, _) => c,
    };
    let invert_match = args.invert_match;
    let only_matching = args.only_matching;
    let search_text = args.text;
    let files_without_match = args.files_without_match;
    let files_with_matches = args.files_with_matches;
    let globs = build_glob_filter(&args.globs)?;

    let hits_mutex: Mutex<Vec<GrepHit>> = Mutex::new(Vec::new());
    let hit_counter = AtomicUsize::new(0);
    let file_counter = AtomicUsize::new(0);
    let stop = std::sync::atomic::AtomicBool::new(false);

    let threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(2)
        .min(8);

    // rg parity: --hidden disables the hidden-file filter, --no-ignore
    // disables gitignore/.ignore handling. `standard_filters(true)` is
    // equivalent to hidden=false + ignore=true + git-ignore=true, so expand
    // it explicitly to honor the flags.
    let walker = {
        let mut b = ignore::WalkBuilder::new(root);
        b.hidden(!args.hidden)
            .git_ignore(!args.no_ignore)
            .git_exclude(!args.no_ignore)
            .git_global(!args.no_ignore)
            .ignore(!args.no_ignore)
            .follow_links(false)
            .threads(threads);
        b.build_parallel()
    };
    walker.run(|| {
        let matcher = &matcher;
        let hits_mutex = &hits_mutex;
        let hit_counter = &hit_counter;
        let file_counter = &file_counter;
        let stop = &stop;
        let candidate_paths = &candidate_paths;
        let bigram_idx = &bigram_idx;
        let bigram_ordinals = &bigram_ordinals;
        let globs = &globs;
        let root = root.to_path_buf();
        let (take, max_count) = (take, max_count);
        let (before_context, after_context) = (before_context, after_context);
        let (invert_match, only_matching, search_text) = (invert_match, only_matching, search_text);
        let (files_with_matches, files_without_match) = (files_with_matches, files_without_match);
        Box::new(move |entry| {
            if stop.load(Ordering::Relaxed) {
                return ignore::WalkState::Quit;
            }
            let Ok(e) = entry else {
                return ignore::WalkState::Continue;
            };
            if !e.file_type().is_some_and(|t| t.is_file()) {
                return ignore::WalkState::Continue;
            }
            // Count every file we visit (only used when there's no cache, to
            // derive the bug-18 denominator without a second walk).
            file_counter.fetch_add(1, Ordering::Relaxed);
            // Grab metadata from the DirEntry *before* consuming it via
            // into_path() — on Windows this is served from the walker's
            // cached FindNextFile attributes, not a fresh stat, so the
            // bigram freshness check below costs no extra syscall. Only
            // fetched when a bigram prefilter is actually active.
            let entry_metadata = if candidate_paths.is_some() {
                e.metadata().ok()
            } else {
                None
            };
            let path = e.into_path();
            // -g include/exclude filter (rg semantics) before any I/O.
            if let Some(gf) = globs {
                if !gf.matches(&root, &path) {
                    return ignore::WalkState::Continue;
                }
            }
            // Bigram prefilter: skip a file only when it is (a) not a
            // candidate AND (b) provably unchanged since indexing. A file
            // added or edited since indexing (even in place) may contain the
            // needle, so it is force-scanned — this is what keeps the prefilter
            // free of false negatives.
            if let Some(cands) = candidate_paths {
                let current = match (bigram_idx, bigram_ordinals, &entry_metadata) {
                    (Some(idx), Some(ord), Some(meta)) => {
                        idx.is_current_with_metadata(&path, ord, meta)
                    }
                    (Some(idx), Some(ord), None) => idx.is_current_at(&path, ord),
                    _ => false,
                };
                if !cands.contains(&path) && current {
                    return ignore::WalkState::Continue;
                }
            }

            let Ok(content) = read_for_search(&path) else {
                return ignore::WalkState::Continue;
            };

            // Binary heuristic (rg skips NUL-containing files unless -a/--text).
            let probe = &content[..content.len().min(8 * 1024)];
            if !search_text && probe.contains(&0u8) {
                return ignore::WalkState::Continue;
            }

            // files-with-matches / files-without-match: only need IF it matches.
            if files_with_matches || files_without_match {
                let matched = matcher.is_match(&content);
                let emit = if files_without_match {
                    !matched
                } else {
                    matched
                };
                if invert_match {
                    // -v inverts the -l/--files-without-match sense too.
                    let emit = !emit;
                    if emit {
                        push_path_hit(&path, hits_mutex, hit_counter, stop, take);
                    }
                } else if emit {
                    push_path_hit(&path, hits_mutex, hit_counter, stop, take);
                }
                return ignore::WalkState::Continue;
            }

            // Newline index built once per file (rg builds its line table
            // once per buffer too — rebuilding per match is O(matches × file)).
            // Skipped outright for files-with-matches mode, which never
            // resolves match offsets to lines — one less scan over the bytes.
            let newline_index: Option<NewlineIndex> =
                if files_with_matches || files_without_match {
                    None
                } else {
                    Some(NewlineIndex::build(&content))
                };
            // `line_hits` holds one entry per emitted display row: normally
            // one row per line (matches merged), but `-o` needs one row per
            // *match* even when several land on the same line, so a flat Vec
            // (sorted by line at the end) replaces the old by-line map, which
            // could only hold a single row per line.
            let mut line_hits: Vec<LineHit> = Vec::new();
            let newline_index = newline_index
                .as_ref()
                .expect("built above whenever a line-resolving branch runs");
            if invert_match {
                // -v: every NON-matching line is a hit (rg semantics).
                let matched_lines: std::collections::HashSet<u32> = matcher
                    .find_iter(&content)
                    .map(|(off, _)| newline_index.byte_to_line(&content, off).0)
                    .collect();
                let text = String::from_utf8_lossy(&content);
                for (idx, line_text) in text.lines().enumerate() {
                    let line = (idx + 1) as u32;
                    if matched_lines.contains(&line) {
                        continue;
                    }
                    line_hits.push((line, line_text.to_string(), Vec::new()));
                    if line_hits.len() >= max_count {
                        break;
                    }
                }
            } else if only_matching {
                // -o: one row per match, showing only the matched text.
                for (per_file, (off, end)) in matcher.find_iter(&content).enumerate() {
                    if per_file >= max_count {
                        break;
                    }
                    let (line, _, _) = newline_index.byte_to_line(&content, off);
                    let snippet = &content[off.min(content.len())..end.min(content.len())];
                    let text = String::from_utf8_lossy(snippet).replace('\n', "\\n");
                    let range = if text.is_empty() {
                        Vec::new()
                    } else {
                        vec![(0, text.len() as u32)]
                    };
                    line_hits.push((line, text, range));
                }
            } else {
                // Default: merge same-line matches into one row carrying all
                // their ranges, keyed by line number.
                let mut by_line: std::collections::BTreeMap<u32, (String, Vec<(u32, u32)>)> =
                    std::collections::BTreeMap::new();
                for (per_file, (off, end)) in matcher.find_iter(&content).enumerate() {
                    if per_file >= max_count {
                        break;
                    }
                    let (line, line_start, slice) = newline_index.byte_to_line(&content, off);
                    // Bug 16: multiline match → render the whole span.
                    let (text, range) = if end > off
                        && end <= content.len()
                        && content[off..end].contains(&b'\n')
                    {
                        let snippet = &content[off..end];
                        let text = String::from_utf8_lossy(snippet).replace('\n', "\\n");
                        let range = if text.is_empty() {
                            None
                        } else {
                            Some((0, text.len() as u32))
                        };
                        (text, range)
                    } else {
                        let text = String::from_utf8_lossy(slice).into_owned();
                        let s = off.saturating_sub(line_start);
                        let e = end.saturating_sub(line_start);
                        let len = text.len() as u32;
                        let range = if e > s {
                            Some((s.min(len as usize) as u32, e.min(len as usize) as u32))
                        } else {
                            None
                        };
                        (text, range)
                    };
                    let entry = by_line.entry(line).or_insert_with(|| (text, Vec::new()));
                    if let Some(r) = range {
                        entry.1.push(r);
                    }
                }
                line_hits = by_line
                    .into_iter()
                    .map(|(line, (text, ranges))| (line, text, ranges))
                    .collect();
            }

            if line_hits.is_empty() {
                return ignore::WalkState::Continue;
            }

            // Only split into a line vec when context was actually requested —
            // it's an extra pass over the file that most callers don't need.
            let file_lines: Vec<&str> = if before_context > 0 || after_context > 0 {
                std::str::from_utf8(&content)
                    .map(|s| s.lines().collect())
                    .unwrap_or_default()
            } else {
                Vec::new()
            };
            let context_slice =
                |line: u32, before: usize, after: usize| -> (Vec<String>, Vec<String>) {
                    if file_lines.is_empty() {
                        return (Vec::new(), Vec::new());
                    }
                    let idx = line.saturating_sub(1) as usize; // 0-based
                    let start = idx.saturating_sub(before);
                    let ctx_before = file_lines[start..idx]
                        .iter()
                        .map(|s| s.to_string())
                        .collect();
                    let end = (idx + 1 + after).min(file_lines.len());
                    let ctx_after = file_lines[(idx + 1).min(file_lines.len())..end]
                        .iter()
                        .map(|s| s.to_string())
                        .collect();
                    (ctx_before, ctx_after)
                };

            let local_hits: Vec<GrepHit> = line_hits
                .into_iter()
                .map(|(line, text, match_ranges)| {
                    let (context_before, context_after) =
                        context_slice(line, before_context, after_context);
                    GrepHit {
                        path: path.to_string_lossy().into_owned(),
                        line,
                        text,
                        match_ranges,
                        context_before,
                        context_after,
                    }
                })
                .collect();

            let prior = hit_counter.fetch_add(local_hits.len(), Ordering::Relaxed);
            if prior >= take {
                stop.store(true, Ordering::Relaxed);
                return ignore::WalkState::Quit;
            }

            if let Ok(mut guard) = hits_mutex.lock() {
                for h in local_hits {
                    if guard.len() >= take {
                        stop.store(true, Ordering::Relaxed);
                        return ignore::WalkState::Quit;
                    }
                    guard.push(h);
                }
            }
            ignore::WalkState::Continue
        })
    });

    let mut hits = hits_mutex.into_inner().unwrap_or_default();
    hits.sort_by(|a, b| a.path.cmp(&b.path).then(a.line.cmp(&b.line)));
    // We stopped scanning at `take` (= offset + limit), so anything found up
    // to that cutoff is either the exact total (fewer than `take`) or a lower
    // bound on it (exactly `take`, meaning more may exist past the cutoff).
    hits.truncate(take);

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
                match_ranges: Vec::new(),
                context_before: Vec::new(),
                context_after: Vec::new(),
            })
            .collect();
    }

    // `total_matches_at_least` / `truncated` describe what we know *before*
    // slicing off `--offset`: how many matches existed up to the `take`
    // cutoff. `truncated` is true when we hit that cutoff exactly, meaning
    // there may be more beyond it that we deliberately didn't scan for.
    let total_matches_at_least = hits.len();
    let truncated = hits.len() >= take;
    if offset > 0 {
        hits = if offset >= hits.len() {
            Vec::new()
        } else {
            hits.split_off(offset)
        };
    }

    // When --group is set, emit symbol-grouped output instead
    if args.group {
        let grouped = build_grouped_result(&args.needle, &hits, mode);
        return super::emit(format, &grouped, |p| {
            let mut out = String::new();
            if p.files.is_empty() {
                out.push_str(&format!("[no matches across {total_files} files]\n"));
                return out;
            }
            for f in &p.files {
                out.push_str(&format!(
                    "{} ({} matches, {} symbols)\n",
                    f.path, f.total_matches, f.total_symbols
                ));
                for g in &f.groups {
                    out.push_str(&format!(
                        "  {} {} @ L{}-L{}\n",
                        g.kind, g.name, g.start_line, g.end_line
                    ));
                    for m in &g.matches {
                        out.push_str(&format!("    - L{} {}\n", m.line, m.text));
                    }
                }
                out.push('\n');
            }
            out
        });
    }

    // When there's no bigram cache, `total_files` was set to usize::MAX and we
    // derived the real count from `file_counter` during the fused walk.
    let total_files = if total_files == usize::MAX {
        file_counter.load(Ordering::Relaxed)
    } else {
        total_files
    };

    let returned = hits.len();
    let payload = GrepResult {
        needle: args.needle,
        hits,
        total_files_searched: total_files,
        mode,
        offset,
        truncated,
        total_matches_at_least,
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
                for c in &h.context_before {
                    out.push_str(&super::render::colorize(&h.path, &path_spec));
                    out.push('-');
                    out.push_str(c);
                    out.push('\n');
                }
                out.push_str(&super::render::colorize(&h.path, &path_spec));
                out.push(':');
                out.push_str(&super::render::colorize(&h.line.to_string(), &line_spec));
                out.push_str(": ");
                out.push_str(&super::render::colorize_matches(&h.text, &h.match_ranges));
                out.push('\n');
                for c in &h.context_after {
                    out.push_str(&super::render::colorize(&h.path, &path_spec));
                    out.push('-');
                    out.push_str(c);
                    out.push('\n');
                }
                if !h.context_before.is_empty() || !h.context_after.is_empty() {
                    out.push_str("--\n");
                }
            }
        }
        if p.hits.is_empty() {
            out.push_str(&format!(
                "[no matches across {} files]\n",
                p.total_files_searched
            ));
        } else {
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

/* ─── Grouped output (--group flag) ─── */

/// A match grouped by its enclosing symbol.
#[derive(Debug, Serialize)]
struct GroupedMatch {
    line: u32,
    text: String,
}

/// A symbol group containing matches.
#[derive(Debug, Serialize)]
struct MatchGroup {
    kind: String,
    name: String,
    start_line: u32,
    end_line: u32,
    matches: Vec<GroupedMatch>,
}

/// Matches in a single file, with symbol groups.
#[derive(Debug, Serialize)]
struct FileGroup {
    path: String,
    total_matches: usize,
    total_symbols: usize,
    groups: Vec<MatchGroup>,
}

/// Enriched grep result with symbol-grouped output.
#[derive(Debug, Serialize)]
struct GroupedGrepResult {
    needle: String,
    total_files: usize,
    total_matches: usize,
    mode: &'static str,
    files: Vec<FileGroup>,
    schema: &'static str,
}

fn build_grouped_result(needle: &str, hits: &[GrepHit], mode: &'static str) -> GroupedGrepResult {
    // Group hits by file
    let mut by_file: std::collections::BTreeMap<String, Vec<&GrepHit>> =
        std::collections::BTreeMap::new();
    for h in hits {
        by_file.entry(h.path.clone()).or_default().push(h);
    }

    let mut files: Vec<FileGroup> = Vec::new();
    for (path, file_hits) in &by_file {
        // Try to parse the file outline for symbol grouping
        let content = ffs_search::bom::read_file(path).ok();
        let entries = content
            .as_deref()
            .map(get_simple_outline)
            .unwrap_or_default();

        let mut groups: Vec<MatchGroup> = Vec::new();
        let mut unmatched: Vec<GroupedMatch> = Vec::new();

        for hit in file_hits {
            let line = hit.line as usize;
            // Find enclosing symbol
            let enclosing = entries
                .iter()
                .find(|e| e.start_line <= line && line <= e.end_line);
            if let Some(sym) = enclosing {
                // Check if we already have a group for this symbol
                if let Some(g) = groups
                    .iter_mut()
                    .find(|g: &&mut MatchGroup| g.name == sym.name && g.kind == sym.kind)
                {
                    g.matches.push(GroupedMatch {
                        line: hit.line,
                        text: hit.text.clone(),
                    });
                } else {
                    groups.push(MatchGroup {
                        kind: sym.kind.clone(),
                        name: sym.name.clone(),
                        start_line: sym.start_line as u32,
                        end_line: sym.end_line as u32,
                        matches: vec![GroupedMatch {
                            line: hit.line,
                            text: hit.text.clone(),
                        }],
                    });
                }
            } else {
                unmatched.push(GroupedMatch {
                    line: hit.line,
                    text: hit.text.clone(),
                });
            }
        }

        // Put unmatched hits in a file-scope group
        if !unmatched.is_empty() {
            groups.push(MatchGroup {
                kind: "file".to_string(),
                name: "<file scope>".to_string(),
                start_line: 0,
                end_line: 0,
                matches: unmatched,
            });
        }

        files.push(FileGroup {
            path: path.clone(),
            total_matches: file_hits.len(),
            total_symbols: entries.len(),
            groups,
        });
    }

    GroupedGrepResult {
        needle: needle.to_string(),
        total_files: files.len(),
        total_matches: hits.len(),
        mode,
        files,
        schema: "v2_grouped",
    }
}

/// A simple structure item for grouping.
struct SymEntry {
    kind: String,
    name: String,
    start_line: usize,
    end_line: usize,
}

/// Get a simple outline from file content using regex-based parsing
/// (lightweight alternative to full tree-sitter outline).
fn get_simple_outline(text: &str) -> Vec<SymEntry> {
    let mut entries = Vec::new();
    let lines: Vec<&str> = text.lines().collect();

    // Detect language from shebang or common patterns
    let lang = if text.contains("fn ") && text.contains("struct ") && text.contains("impl ") {
        "rust"
    } else if text.contains("function ") || text.contains("const ") || text.contains("import ") {
        "typescript"
    } else {
        "generic"
    };

    for (i, line) in lines.iter().enumerate() {
        let line_num = i + 1;
        let trimmed = line.trim_start();

        match lang {
            "rust" => {
                // Functions: pub fn name(...
                if let Some(name) = parse_after_keyword(trimmed, "fn ") {
                    let end = find_block_end(&lines[i..], line_num);
                    entries.push(SymEntry {
                        kind: "function".into(),
                        name,
                        start_line: line_num,
                        end_line: end,
                    });
                }
                // Structs: struct Name { ...
                else if let Some(name) = parse_after_keyword(trimmed, "struct ") {
                    let end = find_block_end(&lines[i..], line_num);
                    entries.push(SymEntry {
                        kind: "struct".into(),
                        name,
                        start_line: line_num,
                        end_line: end,
                    });
                }
                // Enums: enum Name { ...
                else if let Some(name) = parse_after_keyword(trimmed, "enum ") {
                    let end = find_block_end(&lines[i..], line_num);
                    entries.push(SymEntry {
                        kind: "enum".into(),
                        name,
                        start_line: line_num,
                        end_line: end,
                    });
                }
                // Traits: trait Name { ...
                else if let Some(name) = parse_after_keyword(trimmed, "trait ") {
                    let end = find_block_end(&lines[i..], line_num);
                    entries.push(SymEntry {
                        kind: "trait".into(),
                        name,
                        start_line: line_num,
                        end_line: end,
                    });
                }
                // impl blocks
                else if let Some(name) = parse_after_keyword(trimmed, "impl ") {
                    // Extract just the type name (before the { or where)
                    let name = name.split(['{', 'w']).next().unwrap_or(&name).trim();
                    let end = find_block_end(&lines[i..], line_num);
                    entries.push(SymEntry {
                        kind: "impl".into(),
                        name: name.to_string(),
                        start_line: line_num,
                        end_line: end,
                    });
                }
            }
            "typescript" => {
                if let Some(name) = parse_after_keyword(trimmed, "function ") {
                    let end = find_ts_block_end(&lines[i..], line_num);
                    entries.push(SymEntry {
                        kind: "function".into(),
                        name,
                        start_line: line_num,
                        end_line: end,
                    });
                } else if let Some(name) = parse_after_keyword(trimmed, "class ") {
                    let end = find_ts_block_end(&lines[i..], line_num);
                    entries.push(SymEntry {
                        kind: "class".into(),
                        name,
                        start_line: line_num,
                        end_line: end,
                    });
                } else if let Some(name) = parse_after_keyword(trimmed, "interface ") {
                    let end = find_ts_block_end(&lines[i..], line_num);
                    entries.push(SymEntry {
                        kind: "interface".into(),
                        name,
                        start_line: line_num,
                        end_line: end,
                    });
                }
            }
            "generic" => {
                // Generic function detection for any language
                for kw in &["fn ", "def ", "func ", "function "] {
                    if let Some(name) = parse_after_keyword(trimmed, kw) {
                        entries.push(SymEntry {
                            kind: "definition".into(),
                            name,
                            start_line: line_num,
                            end_line: line_num + 5,
                        });
                        break;
                    }
                }
            }
            _ => {}
        }
    }

    // Merge overlapping entries
    entries.sort_by_key(|a| a.start_line);
    entries
}

fn parse_after_keyword(line: &str, kw: &str) -> Option<String> {
    if !line.starts_with(kw) {
        // Also check with pub/export prefix
        let pub_prefixes = ["pub ", "pub(crate) ", "pub(super) ", "export "];
        for prefix in &pub_prefixes {
            if line.starts_with(prefix) {
                let after_prefix = line.strip_prefix(prefix)?;
                if after_prefix.starts_with(kw) {
                    return parse_after_keyword(after_prefix, kw);
                }
            }
        }
        return None;
    }
    let rest = line.strip_prefix(kw)?;
    // Extract name (up to (, <, :, {, whitespace)
    let name = rest.split(['(', '<', ':', '{', ' ']).next()?.trim();
    if name.is_empty() {
        None
    } else {
        Some(name.to_string())
    }
}

fn find_block_end(lines: &[&str], start: usize) -> usize {
    let mut depth: i32 = 0;
    let mut first_brace = false;
    for (i, line) in lines.iter().enumerate() {
        let abs_line = start + i;
        for &b in line.as_bytes() {
            if b == b'{' {
                depth += 1;
                first_brace = true;
            } else if b == b'}' {
                depth -= 1;
            }
        }
        if first_brace && depth <= 0 {
            return abs_line;
        }
    }
    start + lines.len()
}

fn find_ts_block_end(lines: &[&str], start: usize) -> usize {
    find_block_end(lines, start)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_regex_metachars() {
        assert!(looks_like_regex("foo.*bar"));
        assert!(looks_like_regex("^EXPORT"));
        assert!(looks_like_regex("a|b"));
        assert!(!looks_like_regex("EXPORT_SYMBOL_GPL"));
        assert!(!looks_like_regex("simple_word"));
    }

    #[test]
    fn byte_to_line_basic() {
        let h = b"first\nsecond\nthird\n";
        let idx = NewlineIndex::build(h);
        assert_eq!(idx.byte_to_line(h, 0).0, 1);
        assert_eq!(idx.byte_to_line(h, 6).0, 2);
        assert_eq!(idx.byte_to_line(h, 13).0, 3);
        // line_start follows the last newline before the offset
        assert_eq!(idx.byte_to_line(h, 6).1, 6);
        assert_eq!(idx.byte_to_line(h, 13).1, 13);
    }

    // ── find_in_chunk ───────────────────────────────────────────────────

    #[test]
    fn find_in_chunk_literal_match() {
        assert!(find_in_chunk(b"hello world", b"world", false));
        assert!(find_in_chunk(b"hello world", b"hello", false));
        assert!(!find_in_chunk(b"hello world", b"xyz", false));
    }

    #[test]
    fn find_in_chunk_empty_needle() {
        assert!(find_in_chunk(b"anything", b"", false));
    }

    #[test]
    fn find_in_chunk_case_insensitive() {
        assert!(find_in_chunk(b"Hello World", b"hello", true));
        assert!(find_in_chunk(b"Hello World", b"WORLD", true));
        assert!(!find_in_chunk(b"Hello World", b"xyz", true));
    }

    #[test]
    fn find_in_chunk_single_byte() {
        assert!(find_in_chunk(b"abcdef", b"d", false));
        assert!(!find_in_chunk(b"abcdef", b"z", false));
    }

    // ── stream_match_literal ────────────────────────────────────────────

    #[test]
    fn stream_match_basic() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.txt");
        std::fs::write(&path, b"line one\nline two\nline three\n").unwrap();
        assert!(stream_match_literal(&path, b"two", false));
        assert!(!stream_match_literal(&path, b"four", false));
    }

    #[test]
    fn stream_match_empty_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("empty.txt");
        std::fs::write(&path, b"").unwrap();
        assert!(!stream_match_literal(&path, b"anything", false));
    }

    #[test]
    fn stream_match_cross_boundary() {
        // Needle spans two 64KB chunks — verify the overlap logic works.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("boundary.txt");
        // Place "NEED" at end of first chunk, "LE" at start of second.
        let mut data = vec![b'a'; 64 * 1024];
        let pos = 64 * 1024 - 4;
        data[pos..pos + 4].copy_from_slice(b"NEED");
        data.extend_from_slice(b"LEmore content here");
        std::fs::write(&path, &data).unwrap();
        assert!(stream_match_literal(&path, b"NEEDLE", false));
        assert!(!stream_match_literal(&path, b"NEEDLENOPE", false));
    }

    #[test]
    fn stream_match_case_insensitive() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("case.txt");
        std::fs::write(&path, b"Hello World").unwrap();
        assert!(stream_match_literal(&path, b"hello", true));
        assert!(stream_match_literal(&path, b"WORLD", true));
        assert!(!stream_match_literal(&path, b"xyz", true));
    }

    #[test]
    fn stream_match_missing_file() {
        let path = std::path::PathBuf::from("/nonexistent/file.txt");
        assert!(!stream_match_literal(&path, b"test", false));
    }

    #[test]
    fn stream_match_large_file_first_match() {
        // Verify streaming stops early (doesn't read entire 1MB file).
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("large.txt");
        let mut data = vec![b'x'; 1024 * 1024]; // 1 MB
        data[100] = b'Y';
        data[101] = b'Z';
        std::fs::write(&path, &data).unwrap();
        assert!(stream_match_literal(&path, b"YZ", false));
    }
}
