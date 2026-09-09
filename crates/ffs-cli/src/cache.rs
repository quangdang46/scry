//! On-disk cache for the tree-sitter `SymbolIndex`.
//!
//! Layout under `<repo-root>/.ffs/`:
//!
//! ```text
//! .ffs/
//!   meta.json                    -> CacheMeta (schema, version, git head, …)
//!   symbol_index.postcard.zst    -> postcard(SymbolIndexSnapshot) | zstd
//! ```
//!
//! Invalidation strategy: the cache is treated as fresh when the schema
//! version, git HEAD, and file count all match. Anything else triggers a
//! full rebuild on the next code-navigation command.

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use ffs_engine::{Engine, EngineConfig};
use ffs_symbol::{SymbolIndex, SymbolIndexSnapshot};

use crate::bigram::GrepBigram;

const SCHEMA_VERSION: &str = "v1";
const FFS_VERSION: &str = env!("CARGO_PKG_VERSION");
const META_FILE: &str = "meta.json";
const SYMBOL_FILE: &str = "symbol_index.postcard.zst";
const BIGRAM_FILE: &str = "bigram.postcard.zst";
const ZSTD_LEVEL: i32 = 3;

/// Persisted alongside each cache payload — used to decide whether the cache
/// can be reused on a subsequent run.
#[derive(Debug, Serialize, Deserialize)]
struct CacheMeta {
    schema_version: String,
    ffs_version: String,
    git_head: Option<String>,
    file_count: usize,
    generated_at_ms: u128,
}

/// Handle to a `<root>/.ffs/` directory. Construction is cheap and infallible.
pub struct CacheDir {
    dir: PathBuf,
}

impl CacheDir {
    /// Locate the cache directory for `root`. Does not touch the filesystem.
    #[must_use]
    pub fn at(root: &Path) -> Self {
        Self {
            dir: root.join(".ffs"),
        }
    }

    /// Create `<root>/.ffs/` if it does not already exist.
    pub fn ensure(&self) -> Result<()> {
        fs::create_dir_all(&self.dir).with_context(|| format!("create {:?}", self.dir))
    }

    /// Path to the symbol-index payload.
    #[must_use]
    pub fn symbol_path(&self) -> PathBuf {
        self.dir.join(SYMBOL_FILE)
    }

    /// Path to the meta.json file.
    #[must_use]
    pub fn meta_path(&self) -> PathBuf {
        self.dir.join(META_FILE)
    }

    /// Path to the bigram-index payload.
    #[must_use]
    pub fn bigram_path(&self) -> PathBuf {
        self.dir.join(BIGRAM_FILE)
    }

    /// Load the cached symbol index for `root`, returning `None` if the cache
    /// is missing, corrupt, or invalidated by a metadata mismatch.
    pub fn load_symbol_index(&self, root: &Path) -> Option<SymbolIndex> {
        let meta = self.read_meta().ok()?;
        if meta.schema_version != SCHEMA_VERSION {
            return None;
        }
        if !head_matches(root, meta.git_head.as_deref()) {
            return None;
        }
        if !file_count_within_tolerance(root, meta.file_count) {
            return None;
        }
        self.read_payload()
    }

    /// Like `load_symbol_index`, but only enforces that the on-disk schema
    /// version matches. Returns `Some` even when git HEAD or file count
    /// drifted — used by the incremental-refresh path which can re-parse
    /// changed files instead of rebuilding from scratch.
    pub fn load_symbol_index_stale(&self, _root: &Path) -> Option<SymbolIndex> {
        let meta = self.read_meta().ok()?;
        if meta.schema_version != SCHEMA_VERSION {
            return None;
        }
        self.read_payload()
    }

    fn read_payload(&self) -> Option<SymbolIndex> {
        let bytes = fs::read(self.symbol_path()).ok()?;
        let decompressed = zstd::stream::decode_all(&bytes[..]).ok()?;
        let snap: SymbolIndexSnapshot = postcard::from_bytes(&decompressed).ok()?;
        Some(SymbolIndex::from_snapshot(snap))
    }

    /// Persist the bigram filter to `<root>/.ffs/bigram.postcard.zst`.
    /// Atomic; uses the same postcard+zstd format as the symbol cache.
    pub fn write_bigram_index(&self, idx: &GrepBigram) -> Result<()> {
        self.ensure()?;
        let payload = postcard::to_allocvec(idx).context("postcard serialize bigram index")?;
        let mut compressed = Vec::with_capacity(payload.len() / 4);
        zstd::stream::copy_encode(&payload[..], &mut compressed, ZSTD_LEVEL)
            .context("zstd compress bigram index")?;
        atomic_write(&self.bigram_path(), &compressed)
    }

    /// Below this file count, a fused walk+scan of every file is already as
    /// fast as (or faster than) loading, decompressing, and deserializing the
    /// bigram cache — the prefilter only pays for itself on repos large
    /// enough that per-file I/O dominates the walk. Measured on this repo
    /// (~600 files): loading the cache cost ~15% more wall-clock than
    /// skipping it outright for common (non-selective) patterns.
    const BIGRAM_MIN_FILES: usize = 2000;

    /// Load the bigram filter, returning `None` if it's missing, corrupt,
    /// invalidated by a metadata mismatch, or the repo is too small for the
    /// prefilter to be worth its own load cost (see `BIGRAM_MIN_FILES`).
    ///
    /// A matching git HEAD guards against *committed* changes since indexing.
    /// *Uncommitted* changes are handled at search time via per-file
    /// fingerprints: the grep walk force-scans any file whose (mtime, size)
    /// no longer matches the index (see [`GrepBigram::is_current`]), so a
    /// stale index still returns correct results. Returning `None` here just
    /// means "no prefilter — scan every file", which is always correct.
    pub fn load_bigram_index(&self, root: &Path) -> Option<GrepBigram> {
        let meta = self.read_meta().ok()?;
        if meta.schema_version != SCHEMA_VERSION {
            return None;
        }
        if meta.file_count < Self::BIGRAM_MIN_FILES {
            return None;
        }
        if !head_matches(root, meta.git_head.as_deref()) {
            return None;
        }
        let bytes = fs::read(self.bigram_path()).ok()?;
        let decompressed = zstd::stream::decode_all(&bytes[..]).ok()?;
        postcard::from_bytes::<GrepBigram>(&decompressed).ok()
    }

    /// Snapshot `idx` to `<root>/.ffs/symbol_index.postcard.zst`, atomically
    /// replacing any previous payload. Also rewrites `meta.json`.
    pub fn write_symbol_index(&self, idx: &SymbolIndex, root: &Path) -> Result<()> {
        self.ensure()?;
        let snap = idx.snapshot();
        let payload = postcard::to_allocvec(&snap).context("postcard serialize symbol index")?;
        let mut compressed = Vec::with_capacity(payload.len() / 4);
        zstd::stream::copy_encode(&payload[..], &mut compressed, ZSTD_LEVEL)
            .context("zstd compress symbol index")?;
        atomic_write(&self.symbol_path(), &compressed)?;

        let meta = CacheMeta {
            schema_version: SCHEMA_VERSION.to_string(),
            ffs_version: FFS_VERSION.to_string(),
            git_head: read_git_head(root),
            file_count: idx.files_indexed(),
            generated_at_ms: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or(Duration::ZERO)
                .as_millis(),
        };
        let meta_bytes = serde_json::to_vec_pretty(&meta).context("serialize cache meta")?;
        atomic_write(&self.meta_path(), &meta_bytes)?;
        Ok(())
    }

    fn read_meta(&self) -> Result<CacheMeta> {
        let raw = fs::read(self.meta_path()).context("read cache meta")?;
        serde_json::from_slice(&raw).context("parse cache meta")
    }
}

/// Best-effort: build an `Engine` for `root`, reusing the on-disk symbol
/// index when fresh and rebuilding it (plus rewriting the cache) on miss.
///
/// The bloom + outline caches always start empty and rebuild lazily on
/// demand, since the heavy cost being amortized is the tree-sitter parse.
pub fn load_or_build_engine(root: &Path) -> Engine {
    let cache = CacheDir::at(root);
    if let Some(idx) = cache.load_symbol_index(root) {
        return Engine::with_symbols(EngineConfig::default(), Arc::new(idx));
    }
    if let Some(idx) = cache.load_symbol_index_stale(root) {
        // Stale but readable — re-parse only the files that changed.
        refresh_symbol_index(&idx, root);
        if let Err(e) = cache.write_symbol_index(&idx, root) {
            eprintln!("warning: failed to write symbol cache: {}", e);
        }
        return Engine::with_symbols(EngineConfig::default(), Arc::new(idx));
    }
    let engine = Engine::default();
    engine.index(root);
    if let Err(e) = cache.write_symbol_index(&engine.handles.symbols, root) {
        eprintln!("warning: failed to write symbol cache: {}", e);
    }
    engine
}

/// In-place incremental refresh of `idx` against the current contents of
/// `root`. Drops symbols from files that no longer exist and re-parses
/// files whose mtime moved since the snapshot. `SymbolIndex::index_file`
/// is itself mtime-skip aware, so unchanged files stay free.
pub fn refresh_symbol_index(idx: &SymbolIndex, root: &Path) {
    use rayon::prelude::*;

    let on_disk = crate::commands::walk_files(root);
    let on_disk_set: std::collections::HashSet<&Path> =
        on_disk.iter().map(PathBuf::as_path).collect();

    let stale: Vec<PathBuf> = idx
        .indexed_paths()
        .into_iter()
        .filter(|p| !on_disk_set.contains(p.as_path()))
        .collect();
    idx.drop_files(&stale);

    on_disk.par_iter().for_each(|path| {
        let Ok(meta) = std::fs::metadata(path) else {
            return;
        };
        let mtime = meta.modified().unwrap_or(SystemTime::UNIX_EPOCH);
        if idx.mtime_for(path) == Some(mtime) {
            return;
        }
        // 4 MB cap mirrors UnifiedScanner — keeps a single huge file from
        // stalling refresh.
        if meta.len() == 0 || meta.len() > 4 * 1024 * 1024 {
            return;
        }
        let Ok(content) = ffs_search::bom::read_file(path) else {
            return;
        };
        idx.index_file(path, mtime, &content);
    });
}

fn head_matches(root: &Path, expected: Option<&str>) -> bool {
    read_git_head(root).as_deref() == expected
}

/// Allow up to ±5% drift in file count (or ±10 files for tiny repos) before
/// invalidating the cache. This keeps small edits / new tests from forcing a
/// 79-second rebuild on the kernel-sized workspace.
fn file_count_within_tolerance(root: &Path, expected: usize) -> bool {
    let now = crate::commands::walk_files(root).len();
    let tolerance = (expected / 20).max(10) as i64;
    (now as i64 - expected as i64).abs() <= tolerance
}

fn read_git_head(root: &Path) -> Option<String> {
    let head = fs::read_to_string(root.join(".git/HEAD")).ok()?;
    let trimmed = head.trim();
    if let Some(rest) = trimmed.strip_prefix("ref: ") {
        let ref_path = root.join(".git").join(rest);
        fs::read_to_string(ref_path)
            .ok()
            .map(|s| s.trim().to_string())
    } else {
        Some(trimmed.to_string())
    }
}

fn atomic_write(path: &Path, bytes: &[u8]) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).with_context(|| format!("create {parent:?}"))?;
    }
    let tmp = path.with_extension("tmp");
    {
        let mut f = fs::File::create(&tmp).with_context(|| format!("create tmp {tmp:?}"))?;
        f.write_all(bytes).context("write cache payload")?;
        f.sync_all().context("fsync cache payload")?;
    }
    fs::rename(&tmp, path).with_context(|| format!("rename {tmp:?} -> {path:?}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::SystemTime;

    fn build_index(content: &str, name: &str) -> (tempfile::TempDir, SymbolIndex) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(name);
        std::fs::write(&path, content).unwrap();
        let idx = SymbolIndex::new();
        let mtime = std::fs::metadata(&path).unwrap().modified().unwrap();
        idx.index_file(&path, mtime, content);
        (dir, idx)
    }

    #[test]
    fn roundtrips_symbol_index_through_disk() {
        let (dir, idx) = build_index("fn alpha() {}\nfn beta() {}\n", "lib.rs");
        let cache = CacheDir::at(dir.path());
        cache.write_symbol_index(&idx, dir.path()).unwrap();

        let loaded = cache.load_symbol_index(dir.path()).expect("cache hit");
        assert_eq!(loaded.lookup_exact("alpha").len(), 1);
        assert_eq!(loaded.lookup_exact("beta").len(), 1);
        assert_eq!(loaded.symbols_indexed(), 2);
    }

    #[test]
    fn invalidate_on_schema_mismatch() {
        let (dir, idx) = build_index("fn alpha() {}\n", "lib.rs");
        let cache = CacheDir::at(dir.path());
        cache.write_symbol_index(&idx, dir.path()).unwrap();

        // Tamper with meta to bump schema_version.
        let stale = CacheMeta {
            schema_version: "v0".to_string(),
            ffs_version: FFS_VERSION.to_string(),
            git_head: read_git_head(dir.path()),
            file_count: idx.files_indexed(),
            generated_at_ms: 0,
        };
        atomic_write(&cache.meta_path(), &serde_json::to_vec(&stale).unwrap()).unwrap();

        assert!(cache.load_symbol_index(dir.path()).is_none());
    }

    #[test]
    fn invalidate_on_head_change() {
        let (dir, idx) = build_index("fn alpha() {}\n", "lib.rs");
        let cache = CacheDir::at(dir.path());
        cache.write_symbol_index(&idx, dir.path()).unwrap();

        // Forge a fake .git/HEAD pointing at a different commit-ish than
        // whatever was captured at write time (which was None for this temp
        // dir, since .git/ doesn't exist there yet).
        std::fs::create_dir_all(dir.path().join(".git")).unwrap();
        std::fs::write(dir.path().join(".git/HEAD"), "deadbeefdeadbeefdeadbeef\n").unwrap();

        assert!(cache.load_symbol_index(dir.path()).is_none());
    }

    #[test]
    fn cache_dir_paths_are_under_dot_ffs() {
        let cache = CacheDir::at(Path::new("/tmp/example"));
        assert!(cache
            .symbol_path()
            .ends_with(".ffs/symbol_index.postcard.zst"));
        assert!(cache.meta_path().ends_with(".ffs/meta.json"));
    }

    #[test]
    fn load_or_build_engine_writes_cache_on_miss() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("lib.rs"), "fn alpha() {}\n").unwrap();

        // Cold: no cache yet.
        let engine = load_or_build_engine(dir.path());
        assert_eq!(engine.handles.symbols.lookup_exact("alpha").len(), 1);

        // Cache file should now exist.
        let cache = CacheDir::at(dir.path());
        assert!(cache.symbol_path().exists());

        // Warm: re-running should hit the cache (no panic, same answer).
        let engine = load_or_build_engine(dir.path());
        assert_eq!(engine.handles.symbols.lookup_exact("alpha").len(), 1);
        let _ = SystemTime::now();
    }

    #[test]
    fn load_symbol_index_stale_returns_some_when_head_changes() {
        let (dir, idx) = build_index("fn alpha() {}\n", "lib.rs");
        let cache = CacheDir::at(dir.path());
        cache.write_symbol_index(&idx, dir.path()).unwrap();

        // Plant a fake .git/HEAD so strict load fails the head check.
        std::fs::create_dir_all(dir.path().join(".git")).unwrap();
        std::fs::write(dir.path().join(".git/HEAD"), "deadbeef\n").unwrap();

        assert!(cache.load_symbol_index(dir.path()).is_none());
        assert!(cache.load_symbol_index_stale(dir.path()).is_some());
    }

    #[test]
    fn refresh_symbol_index_drops_deleted_files_and_parses_new_ones() {
        let dir = tempfile::tempdir().unwrap();
        let a = dir.path().join("a.rs");
        let b = dir.path().join("b.rs");
        std::fs::write(&a, "fn from_a() {}\n").unwrap();
        std::fs::write(&b, "fn from_b() {}\n").unwrap();

        // Build initial index covering both files.
        let cache = CacheDir::at(dir.path());
        let engine = load_or_build_engine(dir.path());
        assert_eq!(engine.handles.symbols.lookup_exact("from_a").len(), 1);
        assert_eq!(engine.handles.symbols.lookup_exact("from_b").len(), 1);

        // Delete a.rs, add c.rs, modify b.rs.
        std::fs::remove_file(&a).unwrap();
        std::fs::write(dir.path().join("c.rs"), "fn from_c() {}\n").unwrap();
        // Bump mtime by rewriting b.rs with new content.
        std::thread::sleep(std::time::Duration::from_millis(20));
        std::fs::write(&b, "fn from_b_v2() {}\n").unwrap();

        // Force the next call to take the stale path: tamper with HEAD.
        std::fs::create_dir_all(dir.path().join(".git")).unwrap();
        std::fs::write(dir.path().join(".git/HEAD"), "deadbeef\n").unwrap();

        // Should hit the stale-load + refresh branch.
        let engine = load_or_build_engine(dir.path());
        let symbols = &engine.handles.symbols;
        assert!(
            symbols.lookup_exact("from_a").is_empty(),
            "deleted symbol must drop"
        );
        assert!(
            symbols.lookup_exact("from_b").is_empty(),
            "old symbol must drop"
        );
        assert_eq!(symbols.lookup_exact("from_b_v2").len(), 1);
        assert_eq!(symbols.lookup_exact("from_c").len(), 1);

        // Cache file should be present (rewritten by refresh path).
        assert!(cache.symbol_path().exists());
    }

    #[test]
    fn refresh_symbol_index_preserves_unchanged_files() {
        let dir = tempfile::tempdir().unwrap();
        let stable = dir.path().join("stable.rs");
        let touched = dir.path().join("touched.rs");
        std::fs::write(&stable, "fn alpha() {}\n").unwrap();
        std::fs::write(&touched, "fn beta() {}\n").unwrap();

        let engine = load_or_build_engine(dir.path());
        let stable_mtime = engine.handles.symbols.mtime_for(&stable);
        assert!(stable_mtime.is_some());

        // Modify touched.rs only. Force stale path.
        std::thread::sleep(std::time::Duration::from_millis(20));
        std::fs::write(&touched, "fn beta_v2() {}\n").unwrap();
        std::fs::create_dir_all(dir.path().join(".git")).unwrap();
        std::fs::write(dir.path().join(".git/HEAD"), "deadbeef\n").unwrap();

        let engine = load_or_build_engine(dir.path());
        // stable.rs's mtime should be unchanged in the index (no re-parse).
        assert_eq!(engine.handles.symbols.mtime_for(&stable), stable_mtime);
        assert_eq!(engine.handles.symbols.lookup_exact("alpha").len(), 1);
        assert_eq!(engine.handles.symbols.lookup_exact("beta_v2").len(), 1);
    }
}
