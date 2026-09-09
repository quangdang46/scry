//! End-to-end tests for the rg-parity flags added to `ffs grep`:
//! `-i`, `-v`, `-o`, `-g`, `-m`, `--files-without-match`, `--hidden`,
//! `--no-ignore`, `-a`.

use std::path::Path;
use std::process::Command;

use serde_json::Value;
use tempfile::TempDir;

fn binary() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_BIN_EXE_ffs"))
}

fn write_file(dir: &Path, rel: &str, body: &str) {
    let p = dir.join(rel);
    std::fs::create_dir_all(p.parent().unwrap()).unwrap();
    std::fs::write(&p, body).unwrap();
}

fn grep_json(root: &Path, args: &[&str]) -> Value {
    let mut cmd = Command::new(binary());
    cmd.args(["--root", root.to_str().unwrap(), "--format", "json", "grep"]);
    cmd.args(args);
    let out = cmd.output().expect("run ffs grep");
    assert!(
        out.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    serde_json::from_slice(&out.stdout).expect("valid json")
}

#[test]
fn ignore_case_forces_insensitive_over_smart_case() {
    let tmp = TempDir::new().unwrap();
    write_file(tmp.path(), "a.rs", "TODO here\ntodo there\n");
    // Without -i, an uppercase pattern is smart-case sensitive and would
    // only match line 1. With -i it matches both.
    let v = grep_json(tmp.path(), &["-i", "TODO"]);
    assert_eq!(v["hits"].as_array().unwrap().len(), 2);
}

#[test]
fn invert_match_selects_non_matching_lines() {
    let tmp = TempDir::new().unwrap();
    write_file(tmp.path(), "a.rs", "keep this\nfoo line\nkeep that\n");
    let v = grep_json(tmp.path(), &["-v", "foo"]);
    let hits = v["hits"].as_array().unwrap();
    assert_eq!(hits.len(), 2);
    assert_eq!(hits[0]["text"], "keep this");
    assert_eq!(hits[1]["text"], "keep that");
}

#[test]
fn only_matching_emits_one_row_per_match_on_shared_line() {
    let tmp = TempDir::new().unwrap();
    write_file(tmp.path(), "a.rs", "foo bar foo\n");
    let v = grep_json(tmp.path(), &["-o", "foo"]);
    let hits = v["hits"].as_array().unwrap();
    // Two occurrences of "foo" on the same line -> two rows, not one merged row.
    assert_eq!(hits.len(), 2);
    for h in hits {
        assert_eq!(h["text"], "foo");
    }
}

#[test]
fn glob_filter_includes_only_matching_extension() {
    let tmp = TempDir::new().unwrap();
    write_file(tmp.path(), "a.rs", "needle\n");
    write_file(tmp.path(), "b.txt", "needle\n");
    let v = grep_json(tmp.path(), &["-g", "*.rs", "needle"]);
    let hits = v["hits"].as_array().unwrap();
    assert_eq!(hits.len(), 1);
    assert!(hits[0]["path"].as_str().unwrap().ends_with("a.rs"));
}

#[test]
fn glob_filter_exclude_pattern() {
    let tmp = TempDir::new().unwrap();
    write_file(tmp.path(), "a.rs", "needle\n");
    write_file(tmp.path(), "b.rs", "needle\n");
    let v = grep_json(tmp.path(), &["-g", "!b.rs", "needle"]);
    let hits = v["hits"].as_array().unwrap();
    assert_eq!(hits.len(), 1);
    assert!(hits[0]["path"].as_str().unwrap().ends_with("a.rs"));
}

#[test]
fn max_count_per_file_caps_hits_per_file() {
    let tmp = TempDir::new().unwrap();
    write_file(tmp.path(), "a.rs", "needle\nneedle\nneedle\n");
    let v = grep_json(tmp.path(), &["-m", "2", "needle"]);
    assert_eq!(v["hits"].as_array().unwrap().len(), 2);
}

#[test]
fn files_without_match_lists_non_matching_files_only() {
    let tmp = TempDir::new().unwrap();
    write_file(tmp.path(), "a.rs", "needle\n");
    write_file(tmp.path(), "b.rs", "nothing\n");
    let v = grep_json(tmp.path(), &["--files-without-match", "needle"]);
    let hits = v["hits"].as_array().unwrap();
    assert_eq!(hits.len(), 1);
    assert!(hits[0]["path"].as_str().unwrap().ends_with("b.rs"));
}

#[test]
fn hidden_flag_reveals_dotfiles() {
    let tmp = TempDir::new().unwrap();
    write_file(tmp.path(), ".hidden.rs", "needle\n");
    let without = grep_json(tmp.path(), &["needle"]);
    assert_eq!(without["hits"].as_array().unwrap().len(), 0);
    let with_hidden = grep_json(tmp.path(), &["--hidden", "needle"]);
    assert_eq!(with_hidden["hits"].as_array().unwrap().len(), 1);
}
