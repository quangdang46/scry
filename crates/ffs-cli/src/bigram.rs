//! Standalone per-bigram inverted file bitset for `ffs grep` rare patterns.
//!
//! Built during `ffs index` and persisted to
//! `<root>/.ffs/bigram.postcard.zst`. At grep time we extract the
//! pattern's printable-ASCII bigrams, AND their bitsets, and keep only
//! files that survived. False positives are fine — the literal SIMD
//! scan downstream rejects them; false negatives must not happen so
//! we always lowercase both sides and only treat bigrams of printable
//! ASCII (32..=126) characters as discriminators.
//!
//! Sized for typical repos (≤500k files): for each present bigram a
//! `Vec<u64>` of `(file_count + 63) / 64` words. zstd-19 compresses
//! the resulting payload tightly because most bitsets are sparse.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// Maximum file size considered for bigram extraction. Mirrors
/// `UnifiedScanner` so the bigram index covers the same shape of
/// candidates the grep scan would ever read.
const MAX_FILE_BYTES: u64 = 4 * 1024 * 1024;

/// File-change fingerprint `(mtime_secs, len)` used to detect whether a file
/// was modified since it was indexed. A change in either field means the file
/// may contain new bigrams and must be force-scanned at grep time.
pub(crate) type FileFingerprint = (i64, u64);

/// Inverted bigram index. `paths[i]` ↔ bit `i` in every posting bitset.
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct GrepBigram {
    pub paths: Vec<PathBuf>,
    /// `bigram_id (high<<8 | low) -> bitset of length (paths.len()+63)/64`.
    pub posting: HashMap<u16, Vec<u64>>,
    /// Per-file fingerprints `(modified_secs, len)` recorded at index time.
    ///
    /// At grep time the search walk compares each visited file's current
    /// `(mtime, size)` against this entry. Any file whose fingerprint changed
    /// since indexing (uncommitted add/edit — even in-place edits that keep the
    /// same file count) may contain new bigrams, so it is force-scanned even if
    /// it is not in the bigram candidate set. This is what makes the prefilter
    /// *never* produce a false negative without needing a git-level freshness
    /// check per invocation. `paths[i]` ↔ `fingerprints[i]`.
    pub fingerprints: Vec<(i64, u64)>,
}

impl GrepBigram {
    /// Number of files indexed.
    #[must_use]
    pub fn file_count(&self) -> usize {
        self.paths.len()
    }

    /// Number of bigrams with at least one file in their posting list.
    #[must_use]
    pub fn bigram_count(&self) -> usize {
        self.posting.len()
    }

    /// Build an inverted index over `files` by reading each file once and
    /// hashing its 2-byte sliding window. Files that look binary (NUL in
    /// the first 8 KB), exceed the size cap, or fail to read are silently
    /// skipped — they remain in `paths` but contribute no bigrams, so a
    /// query treats them as "unknown content" and they survive prefilter.
    pub fn build(files: &[PathBuf]) -> Self {
        let n = files.len();
        let words = n.div_ceil(64).max(1);

        // Stage 1: per-file bigram set + fingerprint in parallel.
        let per_file: Vec<Option<(HashSet<u16>, FileFingerprint)>> = files
            .par_iter()
            .map(|path| extract_file_bigrams(path))
            .collect();

        // Stage 2: invert into bigram → file-bitset.
        let mut posting: HashMap<u16, Vec<u64>> = HashMap::new();
        let mut fingerprints = Vec::with_capacity(n);
        for (idx, entry) in per_file.into_iter().enumerate() {
            let Some((set, fp)) = entry else {
                // Binary / unreadable / oversized file: no bigrams, unknown
                // content — fingerprint records its state so a later change
                // still force-scans it.
                let fp = file_fingerprint(&files[idx]).unwrap_or((0, 0));
                fingerprints.push(fp);
                continue;
            };
            fingerprints.push(fp);
            let word = idx / 64;
            let bit = 1u64 << (idx % 64);
            for key in set {
                let bs = posting.entry(key).or_insert_with(|| vec![0u64; words]);
                bs[word] |= bit;
            }
        }

        Self {
            paths: files.to_vec(),
            posting,
            fingerprints,
        }
    }

    /// Return the subset of indexed paths that *might* contain `pattern`.
    /// Returns `None` when the pattern has no bigrams to match against
    /// (length < 2 or all bigrams contain non-printable bytes); in that
    /// case the caller should fall back to scanning every file.
    #[must_use]
    pub fn filter(&self, pattern: &[u8]) -> Option<Vec<&Path>> {
        if pattern.len() < 2 || self.paths.is_empty() {
            return None;
        }
        let n = self.paths.len();
        let words = n.div_ceil(64).max(1);
        let mut candidates = vec![u64::MAX; words];
        // Mask off bits past file_count in the last word.
        if !n.is_multiple_of(64) {
            let last = words - 1;
            candidates[last] = (1u64 << (n % 64)) - 1;
        }

        let mut had_bigram = false;
        for w in pattern.windows(2) {
            let a = w[0];
            let b = w[1];
            if (32..=126).contains(&a) && (32..=126).contains(&b) {
                let key = ((a.to_ascii_lowercase() as u16) << 8) | (b.to_ascii_lowercase() as u16);
                match self.posting.get(&key) {
                    Some(bs) => {
                        for (c, x) in candidates.iter_mut().zip(bs.iter()) {
                            *c &= *x;
                        }
                        had_bigram = true;
                    }
                    None => {
                        // Bigram never seen — pattern cannot match anywhere.
                        return Some(Vec::new());
                    }
                }
            }
        }
        if !had_bigram {
            return None;
        }

        // Quick selectivity check: if >80% of files survive on repos with
        // 100+ files, the filter provides little benefit and its overhead
        // (HashSet construction, per-file membership check) outweighs the
        // savings. Return None to tell the caller to scan everything.
        let mut count = 0u64;
        for &w in &candidates {
            count += w.count_ones() as u64;
        }
        if n >= 100 && count as usize > (n * 80) / 100 {
            return None;
        }

        let mut survivors: Vec<&Path> = Vec::with_capacity(count as usize);
        for (idx, path) in self.paths.iter().enumerate() {
            let word = idx / 64;
            if candidates[word] & (1u64 << (idx % 64)) != 0 {
                survivors.push(path.as_path());
            }
        }
        Some(survivors)
    }

    /// Build a path → ordinal map once per grep invocation, so per-file
    /// currency checks stay O(1). The previous linear `position()` scan made
    /// the hot path O(files) per visited file — O(files²) compares on the walk.
    #[must_use]
    pub fn ordinal_map(&self) -> HashMap<&Path, usize> {
        let mut map = HashMap::with_capacity(self.paths.len() * 2);
        for (idx, p) in self.paths.iter().enumerate() {
            map.insert(p.as_path(), idx);
        }
        map
    }

    /// True when `path` is one of the indexed files whose fingerprint
    /// (mtime, size) still matches what was recorded at index time.
    ///
    /// A file that is *not* in the index (added since indexing) or whose
    /// fingerprint changed (edited since indexing — even in place, so the file
    /// count and candidate bitsets are stale) must be treated as "content
    /// unknown": the caller force-scans it regardless of the candidate set.
    /// This guarantees the prefilter never skips a file that could contain the
    /// pattern.
    ///
    /// Prefer [`GrepBigram::is_current_at`] with a prebuilt
    /// [`GrepBigram::ordinal_map`] on hot paths — this convenience wrapper
    /// builds the map per call (O(files)) and is only for tests / cold paths.
    #[must_use]
    #[allow(dead_code)] // test/cold-path convenience wrapper around is_current_at
    pub fn is_current(&self, path: &Path) -> bool {
        let map = self.ordinal_map();
        self.is_current_at(path, &map)
    }

    /// Ordinal-map variant of [`GrepBigram::is_current`]: O(1) map lookup +
    /// one stat, no per-call map build. Pass the map from `ordinal_map()`.
    ///
    /// Prefer [`GrepBigram::is_current_with_metadata`] when the caller
    /// already has `Metadata` in hand (e.g. from a directory walker) — it
    /// avoids this method's extra `stat` syscall per file.
    #[must_use]
    #[allow(dead_code)] // convenience wrapper; hot path uses is_current_with_metadata
    pub fn is_current_at(&self, path: &Path, ordinals: &HashMap<&Path, usize>) -> bool {
        let Some(meta) = std::fs::metadata(path).ok() else {
            return false;
        };
        self.is_current_with_metadata(path, ordinals, &meta)
    }

    /// Like [`GrepBigram::is_current_at`] but takes `Metadata` the caller
    /// already fetched — e.g. `ignore::DirEntry::metadata()`, which on
    /// Windows is served from the `FindNextFile` attributes cached by the
    /// walker rather than a fresh `stat`. Avoiding a second per-file stat
    /// call here matters once the walk visits thousands of files.
    #[must_use]
    pub fn is_current_with_metadata(
        &self,
        path: &Path,
        ordinals: &HashMap<&Path, usize>,
        meta: &std::fs::Metadata,
    ) -> bool {
        let Some(&idx) = ordinals.get(path) else {
            // Not in the index → added since indexing → stale/unknown content.
            return false;
        };
        let Some(recorded) = self.fingerprints.get(idx) else {
            return false;
        };
        fingerprint_from_metadata(meta).as_ref() == Some(recorded)
    }
}

/// Read `(mtime_secs, len)` for a file — the fingerprint used to detect
/// whether a file changed since it was indexed. A change in either field
/// means the file may contain new bigrams and must be force-scanned.
fn file_fingerprint(path: &Path) -> Option<FileFingerprint> {
    let meta = std::fs::metadata(path).ok()?;
    Some(fingerprint_from_metadata(&meta).unwrap_or((0, meta.len())))
}

/// Derive `(mtime_secs, len)` from an already-fetched `Metadata`, without a
/// fresh `stat` call.
fn fingerprint_from_metadata(meta: &std::fs::Metadata) -> Option<FileFingerprint> {
    let mtime = meta
        .modified()
        .ok()
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    Some((mtime, meta.len()))
}

fn extract_file_bigrams(path: &Path) -> Option<(HashSet<u16>, FileFingerprint)> {
    let meta = std::fs::metadata(path).ok()?;
    if !meta.is_file() {
        return None;
    }
    let len = meta.len();
    if len == 0 || len > MAX_FILE_BYTES {
        return None;
    }
    let content = std::fs::read(path).ok()?;
    // Quick binary sniff: NUL in the first 8 KB.
    let probe = &content[..content.len().min(8 * 1024)];
    if probe.contains(&0u8) {
        return None;
    }
    let mut set: HashSet<u16> = HashSet::with_capacity(1024);
    for w in content.windows(2) {
        let a = w[0];
        let b = w[1];
        if (32..=126).contains(&a) && (32..=126).contains(&b) {
            let key = ((a.to_ascii_lowercase() as u16) << 8) | (b.to_ascii_lowercase() as u16);
            set.insert(key);
        }
    }
    Some((set, file_fingerprint(path)?))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write(dir: &Path, rel: &str, body: &str) -> PathBuf {
        let p = dir.join(rel);
        std::fs::create_dir_all(p.parent().unwrap()).unwrap();
        std::fs::write(&p, body).unwrap();
        p
    }

    #[test]
    fn build_and_filter_keeps_files_with_pattern() {
        let dir = tempfile::tempdir().unwrap();
        let a = write(dir.path(), "a.rs", "fn alpha() {}\n");
        let b = write(dir.path(), "b.rs", "fn beta() {}\n");
        let c = write(dir.path(), "c.rs", "fn gamma() {}\n");
        let idx = GrepBigram::build(&[a.clone(), b.clone(), c.clone()]);
        let hits = idx.filter(b"alpha").expect("had bigrams");
        assert!(hits.contains(&a.as_path()));
        assert!(!hits.contains(&b.as_path()));
        assert!(!hits.contains(&c.as_path()));
    }

    #[test]
    fn filter_returns_none_for_too_short_pattern() {
        let dir = tempfile::tempdir().unwrap();
        let a = write(dir.path(), "a.rs", "x");
        let idx = GrepBigram::build(std::slice::from_ref(&a));
        assert!(idx.filter(b"a").is_none());
    }

    #[test]
    fn filter_returns_empty_when_bigram_never_seen() {
        let dir = tempfile::tempdir().unwrap();
        let a = write(dir.path(), "a.rs", "abcdef\n");
        let idx = GrepBigram::build(std::slice::from_ref(&a));
        // "qz" doesn't appear in any indexed file.
        let hits = idx.filter(b"qz").expect("had printable bigrams");
        assert!(hits.is_empty());
    }

    #[test]
    fn filter_is_case_insensitive_at_extraction() {
        let dir = tempfile::tempdir().unwrap();
        let a = write(dir.path(), "a.rs", "fn ALPHA() {}\n");
        let idx = GrepBigram::build(std::slice::from_ref(&a));
        // Lowercase pattern still hits because the file's bigrams were
        // lowercased at extraction time.
        let hits = idx.filter(b"alpha").expect("had bigrams");
        assert!(hits.contains(&a.as_path()));
    }

    #[test]
    fn build_skips_binary_files_silently() {
        let dir = tempfile::tempdir().unwrap();
        let bin = dir.path().join("bin.dat");
        // 1 KB of NULs — looks binary.
        std::fs::write(&bin, vec![0u8; 1024]).unwrap();
        let txt = write(dir.path(), "ok.rs", "hello world\n");
        let idx = GrepBigram::build(&[bin.clone(), txt.clone()]);
        // bin.dat contributes nothing to posting; only txt should turn up.
        let hits = idx.filter(b"hello").expect("had bigrams");
        assert!(hits.contains(&txt.as_path()));
        assert!(!hits.contains(&bin.as_path()));
    }

    #[test]
    fn is_current_false_for_file_added_after_index() {
        // Regression: the prefilter must never skip a file added since
        // indexing — that would be a false negative.
        let dir = tempfile::tempdir().unwrap();
        let a = write(dir.path(), "a.rs", "fn alpha() {}\n");
        let idx = GrepBigram::build(&[a]);
        // New file, not in the index → must NOT be considered current.
        let added = write(dir.path(), "zeta.rs", "fn zeta() {}\n");
        assert!(!idx.is_current(&added));
    }

    #[test]
    fn is_current_false_after_in_place_modify() {
        // An in-place edit (same file, content changed) must be detected even
        // though the file is still in the index — a size/mtime change marks it
        // stale so the caller force-scans it.
        let dir = tempfile::tempdir().unwrap();
        let a = write(dir.path(), "a.rs", "fn alpha() {}\n");
        let idx = GrepBigram::build(&[a.clone()]);
        assert!(idx.is_current(&a));
        // Rewrite with different content (likely different length or mtime).
        std::fs::write(&a, "fn zeta_also_here() {}\n").unwrap();
        // Give the filesystem a moment so mtime granularity can't hide the edit.
        std::thread::sleep(std::time::Duration::from_millis(20));
        assert!(!idx.is_current(&a));
    }

    #[test]
    fn is_current_true_for_unchanged_file() {
        let dir = tempfile::tempdir().unwrap();
        let a = write(dir.path(), "a.rs", "fn alpha() {}\n");
        let idx = GrepBigram::build(&[a.clone()]);
        assert!(idx.is_current(&a));
    }

    #[test]
    fn selectivity_skips_filter_when_most_files_match() {
        // When >80% of files survive the filter on repos with 100+ files,
        // filter() returns None to tell the caller to scan everything.
        let dir = tempfile::tempdir().unwrap();
        let mut files = Vec::new();
        // Create 120 files, all containing "ab" — a very common bigram.
        for i in 0..120 {
            let f = write(dir.path(), &format!("f{i}.rs"), "fn alpha_ab() {}\n");
            files.push(f);
        }
        let idx = GrepBigram::build(&files);
        // "ab" appears in every file → all 120 survive → >80% → returns None.
        let result = idx.filter(b"ab");
        assert!(result.is_none(), "should skip filter when >80% match");
    }

    #[test]
    fn selectivity_keeps_filter_when_few_files_match() {
        // When <80% of files survive, filter() returns Some(candidates).
        let dir = tempfile::tempdir().unwrap();
        let mut files = Vec::new();
        // 100 files: only 10 contain "zz" (a rare bigram).
        for i in 0..100 {
            let content = if i < 10 {
                "fn has_zz_zz() {}\n"
            } else {
                "fn normal() {}\n"
            };
            let f = write(dir.path(), &format!("f{i}.rs"), content);
            files.push(f);
        }
        let idx = GrepBigram::build(&files);
        let result = idx.filter(b"zz");
        let candidates = result.expect("should return candidates");
        assert!(
            candidates.len() <= 10,
            "should filter to ~10 candidates, got {}",
            candidates.len()
        );
    }
}
