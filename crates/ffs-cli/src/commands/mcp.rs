//! Minimal MCP server stub. Speaks JSON-RPC 2.0 over stdio so any MCP-aware
//! client (Claude Code, Cursor, …) can call the same handlers as the CLI.

use std::io::{BufRead, Write};
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use anyhow::Result;
use clap::Parser;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use ffs_budget::{
    apply_preserving_footer, smart_truncate, AggressiveFilter, BudgetSplit, FilterLevel,
    FilterStrategy, MinimalFilter, NoFilter, TruncationOutcome,
};
use ffs_engine::dispatch::DispatchResult;
use ffs_engine::{Engine, EngineConfig};
use ffs_symbol::lang::detect_file_type;
use ffs_symbol::outline::get_outline_entries;
use ffs_symbol::types::{FileType, OutlineEntry};

#[derive(Debug, Parser)]
pub struct Args {
    /// Optional total token budget propagated to `Engine`.
    #[arg(long)]
    pub budget: Option<u64>,

    /// Workspace root to index. Wins over the global `--root` flag and over
    /// auto-detection (`WORKSPACE_FOLDER_PATHS` / `VSCODE_CWD` / cwd).
    #[arg(value_name = "PATH")]
    pub path: Option<PathBuf>,
}

/// Resolve the MCP workspace root when the user did not pass an explicit path.
///
/// Priority:
/// 1. `WORKSPACE_FOLDER_PATHS` — first existing entry (Cursor / VS Code MCP)
/// 2. `VSCODE_CWD` — VS Code sometimes injects this
/// 3. `std::env::current_dir()`
///
/// Multi-root workspaces only use the first existing path; FilePicker is
/// single-root today.
pub fn resolve_default_root() -> PathBuf {
    if let Some(p) = first_existing_workspace_folder() {
        return p;
    }
    if let Ok(cwd) = std::env::var("VSCODE_CWD") {
        let p = PathBuf::from(cwd.trim().trim_matches('"').trim_matches('\''));
        if p.is_dir() {
            return p;
        }
    }
    std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."))
}

fn first_existing_workspace_folder() -> Option<PathBuf> {
    let raw = std::env::var("WORKSPACE_FOLDER_PATHS").ok()?;
    let paths = parse_workspace_folder_paths(&raw);
    if paths.len() > 1 {
        // Single-root only for now; surface multi-root so users aren't surprised.
        eprintln!(
            "ffs mcp: WORKSPACE_FOLDER_PATHS has {} entries; indexing the first existing one only",
            paths.len()
        );
    }
    paths.into_iter().find(|p| p.is_dir())
}

/// Split `WORKSPACE_FOLDER_PATHS` on `;` / newlines only — never on `:`,
/// which is the Windows drive-letter separator.
fn parse_workspace_folder_paths(raw: &str) -> Vec<PathBuf> {
    raw.split([';', '\n', '\r'])
        .map(|s| s.trim().trim_matches('"').trim_matches('\''))
        .filter(|s| !s.is_empty())
        .map(PathBuf::from)
        .collect()
}

struct McpState {
    engine: Engine,
    indexed: bool,
}

impl McpState {
    fn new(engine: Engine) -> Self {
        Self {
            engine,
            indexed: false,
        }
    }

    fn ensure_indexed(&mut self, root: &Path) {
        if !self.indexed {
            self.engine.index(root);
            self.indexed = true;
        }
    }
}

#[derive(Debug, Deserialize)]
struct Request {
    #[serde(default)]
    jsonrpc: String,
    id: Option<Value>,
    method: String,
    #[serde(default)]
    params: Value,
}

#[derive(Debug, Serialize)]
struct Response {
    jsonrpc: &'static str,
    id: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<Value>,
}

pub fn run(args: Args, root: &Path) -> Result<()> {
    let cfg = EngineConfig {
        total_token_budget: args.budget.unwrap_or(25_000),
        ..EngineConfig::default()
    };
    let engine = Engine::new(cfg);
    let mut state = McpState::new(engine);

    let stdin = std::io::stdin();
    let mut out = std::io::stdout().lock();

    for line in stdin.lock().lines() {
        let Ok(line) = line else { break };
        if line.trim().is_empty() {
            continue;
        }
        let req: Request = match serde_json::from_str(&line) {
            Ok(r) => r,
            Err(e) => {
                let resp = Response {
                    jsonrpc: "2.0",
                    id: Value::Null,
                    result: None,
                    error: Some(json!({"code": -32700, "message": format!("parse error: {e}")})),
                };
                writeln!(out, "{}", serde_json::to_string(&resp)?)?;
                continue;
            }
        };
        debug_assert!(req.jsonrpc == "2.0" || req.jsonrpc.is_empty());

        let id = req.id.unwrap_or(Value::Null);
        // JSON-RPC 2.0 notifications have no `id`; don't respond.
        if id.is_null() {
            continue;
        }
        let resp = match handle_method(&mut state, root, &req.method, &req.params) {
            Ok(value) => Response {
                jsonrpc: "2.0",
                id,
                result: Some(value),
                error: None,
            },
            Err(e) => Response {
                jsonrpc: "2.0",
                id,
                error: Some(json!({"code": -32000, "message": e.to_string()})),
                result: None,
            },
        };
        writeln!(out, "{}", serde_json::to_string(&resp)?)?;
    }
    Ok(())
}

fn handle_method(state: &mut McpState, root: &Path, method: &str, params: &Value) -> Result<Value> {
    match method {
        // Notifications per JSON-RPC 2.0: no response sent (caller skips
        // writing to stdout when the id is Null).
        "notifications/initialized" => Ok(Value::Null),
        "initialize" => Ok(json!({
            "protocolVersion": "2024-11-05",
            "serverInfo": {"name": "ffs", "version": env!("CARGO_PKG_VERSION")},
            "capabilities": {"tools": {}}
        })),
        "tools/list" => Ok(json!({ "tools": tools_list() })),
        "tools/call" => {
            let name = params
                .get("name")
                .and_then(Value::as_str)
                .ok_or_else(|| anyhow::anyhow!("missing tool name"))?;
            let arguments = params.get("arguments").cloned().unwrap_or(Value::Null);
            handle_tool(state, root, name, &arguments)
        }
        other => Err(anyhow::anyhow!("unknown method: {other}")),
    }
}

// All 16 tools advertised in the README. Each entry uses an integer-typed
// `maxResults` (instead of `number`) and an explicit object schema.
fn tools_list() -> Value {
    json!([
        tool(
            "ffs_grep",
            "Search file contents (replaces Grep). Plain / regex / fuzzy auto-detect.",
            &["query"],
            json!({
                "query": {"type": "string", "description": "Text to find in file contents."},
                "needle": {"type": "string", "description": "Alias for `query` to match the CLI flag name."},
                "maxResults": {"type": "integer", "minimum": 1, "description": "Maximum matching lines to return."},
                "offset": {"type": "integer", "minimum": 0, "description": "Skip this many matches before starting the page. Use with maxResults to page through large result sets."}
            }),
        ),
        tool(
            "ffs_multi_grep",
            "OR-logic multi-pattern content search via SIMD Aho-Corasick. Accepts `queries` or `patterns`.",
            &[],
            json!({
                "queries": {"type": "array", "items": {"type": "string"}, "description": "Patterns to OR together (legacy name)."},
                "patterns": {"type": "array", "items": {"type": "string"}, "description": "Alias for `queries` (matches engine/agent docs)."},
                "maxResults": {"type": "integer", "minimum": 1, "description": "Maximum matching lines to return."},
                "limit": {"type": "integer", "minimum": 1, "description": "Alias for maxResults."}
            }),
        ),
        tool(
            "ffs_glob",
            "Match files by glob pattern (replaces Glob).",
            &["pattern"],
            json!({
                "pattern": {"type": "string", "description": "Glob pattern, for example src/**/*.rs."},
                "maxResults": {"type": "integer", "minimum": 1, "description": "Maximum matching paths to return."},
                "offset": {"type": "integer", "minimum": 0, "description": "Skip this many matches before starting the page. Use with maxResults to page through large result sets."}
            }),
        ),
        tool(
            "ffs_find",
            "Fuzzy file path search.",
            &["query"],
            json!({
                "query": {"type": "string", "description": "Path substring or fuzzy filename query."},
                "needle": {"type": "string", "description": "Alias for `query` to match the CLI flag name."},
                "maxResults": {"type": "integer", "minimum": 1, "description": "Maximum matching paths to return."}
            }),
        ),
        tool(
            "ffs_read",
            "Read a file with token-budget aware truncation (replaces Read).",
            &["path"],
            json!({
                "path": {"type": "string", "description": "Relative or absolute file path."},
                "maxTokens": {"type": "integer", "minimum": 1, "description": "Token budget for the response."}
            }),
        ),
        tool(
            "ffs_outline",
            "Structural outline of a file (functions, classes, top-level decls).",
            &["path"],
            json!({
                "path": {"type": "string", "description": "Relative or absolute file path."}
            }),
        ),
        tool(
            "ffs_symbol",
            "Look up symbol definitions across the workspace.",
            &["name"],
            json!({
                "name": {"type": "string", "description": "Exact symbol name to look up."},
                "maxResults": {"type": "integer", "minimum": 1, "description": "Maximum matching definitions to return."}
            }),
        ),
        tool(
            "ffs_callers",
            "Find call sites of a symbol.",
            &["name"],
            json!({
                "name": {"type": "string", "description": "Symbol whose callers should be listed."},
                "maxResults": {"type": "integer", "minimum": 1, "description": "Maximum matching call sites to return."}
            }),
        ),
        tool(
            "ffs_callees",
            "List symbols referenced inside the body of a definition.",
            &["name"],
            json!({
                "name": {"type": "string", "description": "Symbol whose body should be inspected."},
                "maxResults": {"type": "integer", "minimum": 1, "description": "Maximum referenced symbols to return."}
            }),
        ),
        tool(
            "ffs_refs",
            "Definitions plus single-hop usages of a symbol.",
            &["name"],
            json!({
                "name": {"type": "string", "description": "Symbol to look up."},
                "maxResults": {"type": "integer", "minimum": 1, "description": "Maximum matching usages to return."}
            }),
        ),
        tool(
            "ffs_flow",
            "Drill-down envelope per definition (def + body + callees + callers).",
            &["name"],
            json!({
                "name": {"type": "string", "description": "Symbol whose call envelope is requested."},
                "maxResults": {"type": "integer", "minimum": 1, "description": "Maximum number of definitions to expand."}
            }),
        ),
        tool(
            "ffs_siblings",
            "Peers of a symbol in its parent scope.",
            &["name"],
            json!({
                "name": {"type": "string", "description": "Target symbol whose siblings should be listed."},
                "maxResults": {"type": "integer", "minimum": 1, "description": "Maximum siblings to return."}
            }),
        ),
        tool(
            "ffs_deps",
            "A file's imports plus the workspace files that depend on it.",
            &["path"],
            json!({
                "path": {"type": "string", "description": "File path to analyse, relative to the root."}
            }),
        ),
        tool(
            "ffs_impact",
            "Rank workspace files by how much they'd be affected if `name` changed.",
            &["name"],
            json!({
                "name": {"type": "string", "description": "Symbol whose impact should be assessed."},
                "maxResults": {"type": "integer", "minimum": 1, "description": "Maximum affected files to return."}
            }),
        ),
        tool(
            "ffs_map",
            "Workspace tree annotated with file count and per-directory token estimate.",
            &[],
            json!({
                "depth": {"type": "integer", "minimum": 1, "description": "Maximum directory depth to render."}
            }),
        ),
        tool(
            "ffs_overview",
            "High-signal repo summary: languages, top-defined symbols, entry-point candidates.",
            &[],
            json!({
                "limit": {"type": "integer", "minimum": 1, "description": "How many top symbols / files to surface."}
            }),
        ),
        tool(
            "ffs_dispatch",
            "Auto-classify a free-form query.",
            &["query"],
            json!({
                "query": {"type": "string", "description": "Free-form search or navigation query."}
            }),
        ),
    ])
}

fn tool(name: &str, description: &str, required: &[&str], properties: Value) -> Value {
    json!({
        "name": name,
        "description": description,
        "inputSchema": object_schema(properties, required),
    })
}

fn object_schema(properties: Value, required: &[&str]) -> Value {
    json!({
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": false
    })
}

fn handle_tool(state: &mut McpState, root: &Path, name: &str, args: &Value) -> Result<Value> {
    match name {
        "ffs_grep" => {
            // Bug 3: accept both `query` (MCP idiom) and `needle` (CLI idiom).
            let query = get_query(args)?;
            let limit = get_limit(args, 20);
            let hits = grep_files(root, query, limit);
            Ok(text_json(serde_json::to_string(&hits)?))
        }
        "ffs_multi_grep" => {
            // Accept `queries` (legacy MCP stub) or `patterns` (engine/agent docs).
            let queries = args
                .get("queries")
                .or_else(|| args.get("patterns"))
                .and_then(Value::as_array)
                .ok_or_else(|| anyhow::anyhow!("missing queries or patterns"))?;
            let limit = get_limit(args, 20);
            let mut all_hits = Vec::new();
            // Prefer a single multi-literal pass via the CLI multi-grep path when
            // all entries are plain strings (same OR semantics as Aho-Corasick).
            let needles: Vec<&str> = queries.iter().filter_map(Value::as_str).collect();
            if !needles.is_empty() {
                let mut hits = multi_grep_files(root, &needles, limit);
                all_hits.append(&mut hits);
            }
            all_hits.truncate(limit);
            Ok(text_json(serde_json::to_string(&all_hits)?))
        }
        "ffs_glob" => {
            let pattern = get_string(args, "pattern")?;
            let limit = get_limit(args, 50);
            let hits = glob_files(root, pattern, limit);
            Ok(text_json(serde_json::to_string(&hits)?))
        }
        "ffs_find" => {
            let query = get_query(args)?;
            let limit = get_limit(args, 50);
            let scopes = [root.to_path_buf()];
            let mut hits = super::find::search_matches(&scopes, query);
            if hits.is_empty() {
                hits = super::find::fuzzy_search_matches(&scopes, query);
            }
            hits.truncate(limit);
            Ok(text_json(serde_json::to_string(&hits)?))
        }
        "ffs_dispatch" => {
            let query = get_query(args)?;
            state.ensure_indexed(root);
            let result = state.engine.dispatch(query, root);
            let summary = match result {
                DispatchResult::Symbol { hits, .. } => {
                    json!({"kind": "symbol", "hits": symbol_locs_to_json(hits)})
                }
                DispatchResult::SymbolGlob { hits, .. } => {
                    json!({"kind": "symbol_glob", "hits": symbol_glob_to_json(hits)})
                }
                DispatchResult::FilePath { path, .. } => {
                    json!({"kind": "file_path", "path": path.to_string_lossy()})
                }
                DispatchResult::Glob { pattern, .. } => {
                    json!({"kind": "glob", "pattern": pattern})
                }
                DispatchResult::ContentFallback { .. } => json!({"kind": "content_fallback"}),
            };
            Ok(text_json(summary.to_string()))
        }
        "ffs_read" => {
            let target = get_string(args, "path")?;
            // `path:line` focuses the structural section containing the line,
            // matching the CLI's `ffs read` behavior (Bug: MCP rejected spans).
            let (path_part, line) = parse_target(target);
            let p = if Path::new(path_part).is_absolute() {
                PathBuf::from(path_part)
            } else {
                root.join(path_part)
            };
            let body = match line {
                Some(l) => read_section_mcp(&state.engine, &p, l)?,
                None => state.engine.read(&p).body,
            };
            Ok(text_json(body))
        }
        "ffs_outline" => {
            let target = get_string(args, "path")?;
            let p = if Path::new(target).is_absolute() {
                PathBuf::from(target)
            } else {
                root.join(target)
            };
            let body = match super::outline::render_agent(&p, target) {
                Ok(b) => b,
                Err(e) => format!("[error: {e}]"),
            };
            Ok(text_json(body))
        }
        "ffs_symbol" => {
            let nm = get_string(args, "name")?;
            let limit = get_limit(args, 50);
            state.ensure_indexed(root);
            let mut hits = state.engine.handles.symbols.lookup_exact(nm);
            hits.truncate(limit);
            Ok(text_json(serde_json::to_string(&symbol_locs_to_json(
                hits,
            ))?))
        }
        "ffs_callers" => {
            let nm = get_string(args, "name")?;
            let limit = get_limit(args, 50);
            state.ensure_indexed(root);
            let hits = collect_callers(state, root, nm, limit);
            Ok(text_json(serde_json::to_string(&hits)?))
        }
        "ffs_callees" => {
            let nm = get_string(args, "name")?;
            let limit = get_limit(args, 50);
            state.ensure_indexed(root);
            let hits = collect_callees(state, root, nm, limit);
            Ok(text_json(serde_json::to_string(&hits)?))
        }
        "ffs_refs" => {
            let nm = get_string(args, "name")?;
            let limit = get_limit(args, 50);
            state.ensure_indexed(root);
            let defs = state.engine.handles.symbols.lookup_exact(nm);
            let usages = collect_callers(state, root, nm, limit);
            Ok(text_json(
                json!({
                    "definitions": symbol_locs_to_json(defs),
                    "usages": usages,
                })
                .to_string(),
            ))
        }
        "ffs_flow" => {
            let nm = get_string(args, "name")?;
            let limit = get_limit(args, 5);
            state.ensure_indexed(root);
            let defs = state.engine.handles.symbols.lookup_exact(nm);
            let cards: Vec<Value> = defs
                .into_iter()
                .take(limit)
                .map(|d| {
                    let body = read_section_excerpt(&d.path, d.line, d.end_line, 60);
                    json!({
                        "path": d.path.to_string_lossy(),
                        "line": d.line,
                        "end_line": d.end_line,
                        "body": body,
                    })
                })
                .collect();
            Ok(text_json(serde_json::to_string(&cards)?))
        }
        "ffs_siblings" => {
            let nm = get_string(args, "name")?;
            let limit = get_limit(args, 50);
            state.ensure_indexed(root);
            let hits = collect_siblings(state, nm, limit);
            Ok(text_json(serde_json::to_string(&hits)?))
        }
        "ffs_deps" => {
            let target = get_string(args, "path")?;
            let p = if Path::new(target).is_absolute() {
                PathBuf::from(target)
            } else {
                root.join(target)
            };
            let imports = list_imports(&p);
            Ok(text_json(serde_json::to_string(&imports)?))
        }
        "ffs_impact" => {
            let nm = get_string(args, "name")?;
            let limit = get_limit(args, 50);
            state.ensure_indexed(root);
            let hits = collect_callers(state, root, nm, limit);
            // Roll up call sites by file as a coarse impact estimate.
            let mut by_file: std::collections::BTreeMap<String, u32> =
                std::collections::BTreeMap::new();
            for h in &hits {
                *by_file.entry(h.path.clone()).or_default() += 1;
            }
            let ranked: Vec<Value> = by_file
                .into_iter()
                .map(|(p, n)| json!({"path": p, "weight": n}))
                .collect();
            Ok(text_json(serde_json::to_string(&ranked)?))
        }
        "ffs_map" => {
            let depth = args
                .get("depth")
                .and_then(Value::as_u64)
                .unwrap_or(2)
                .min(10);
            let lines = render_simple_map(root, depth as usize);
            Ok(text_json(lines.join("\n")))
        }
        "ffs_overview" => {
            let limit = get_limit(args, 20);
            state.ensure_indexed(root);
            let mut langs: std::collections::BTreeMap<&'static str, usize> =
                std::collections::BTreeMap::new();
            for path in super::walk_files(root) {
                if let FileType::Code(lang) = detect_file_type(&path) {
                    *langs.entry(lang_label(&lang)).or_default() += 1;
                }
            }
            let summary = json!({
                "languages": langs.into_iter().collect::<Vec<_>>(),
                "top_symbols_limit": limit,
            });
            Ok(text_json(summary.to_string()))
        }
        other => Err(anyhow::anyhow!("unknown tool: {other}")),
    }
}

// Split `path:line` like the CLI's `ffs read` parser: a trailing positive
// integer suffix selects a line; anything else is a plain path (Windows
// drive letters included).
fn parse_target(target: &str) -> (&str, Option<u32>) {
    if let Some((p, rest)) = target.rsplit_once(':') {
        if let Ok(n) = rest.parse::<u32>() {
            if n > 0 {
                return (p, Some(n));
            }
        }
    }
    (target, None)
}

fn deepest_containing(entries: &[OutlineEntry], line: u32) -> Option<OutlineEntry> {
    let mut best: Option<OutlineEntry> = None;
    for e in entries {
        if e.start_line <= line && line <= e.end_line {
            // Prefer a deeper child if one also contains the line.
            if let Some(child) = deepest_containing(&e.children, line) {
                best = Some(child);
            } else {
                best = Some(e.clone());
            }
        }
    }
    best
}

fn slice_lines(content: &str, start_line: u32, end_line: u32) -> String {
    let start = start_line.saturating_sub(1) as usize;
    let end = end_line as usize;
    let mut out = String::new();
    for (i, line) in content.lines().enumerate() {
        if i >= start && i < end {
            out.push_str(line);
            out.push('\n');
        }
        if i >= end {
            break;
        }
    }
    out
}

fn budgeted(filtered: &str, max_bytes: usize) -> (String, TruncationOutcome) {
    let mut buf = String::new();
    let footer = "[truncated to budget]\n";
    let outcome = if filtered.len() <= max_bytes {
        let (out, oc) = smart_truncate(filtered, max_bytes);
        buf.push_str(&out);
        oc
    } else {
        apply_preserving_footer(&mut buf, max_bytes, footer, |target, budget| {
            let take = filtered.len().min(budget);
            target.push_str(&filtered[..take]);
            take
        })
    };
    (buf, outcome)
}

fn section_body(
    engine: &Engine,
    path: &Path,
    line: u32,
    level: FilterLevel,
    budget: u64,
) -> Result<String> {
    let lang = match detect_file_type(path) {
        FileType::Code(l) => l,
        _ => return Err(anyhow::anyhow!("not a code file: {}", path.display())),
    };
    let content = ffs_search::bom::read_file(path)
        .map_err(|e| anyhow::anyhow!("read {}: {e}", path.display()))?;
    let mtime = std::fs::metadata(path)
        .and_then(|m| m.modified())
        .unwrap_or(SystemTime::UNIX_EPOCH);
    let outline = engine
        .handles
        .outlines
        .get_or_compute(path, mtime, &content, lang);

    let entry = deepest_containing(&outline, line)
        .ok_or_else(|| anyhow::anyhow!("no structural section contains line {line}"))?;

    let slice = slice_lines(&content, entry.start_line, entry.end_line);
    let filter: Box<dyn FilterStrategy> = match level {
        FilterLevel::None => Box::new(NoFilter),
        FilterLevel::Minimal => Box::new(MinimalFilter),
        FilterLevel::Aggressive => Box::new(AggressiveFilter),
    };
    let filtered = filter.apply(&slice);

    let split = BudgetSplit::default_for(budget);
    let body_budget_bytes = (split.body * 4) as usize;
    let max_bytes = body_budget_bytes.min(engine.config.max_bytes_per_result);
    let (body, _outcome) = budgeted(&filtered, max_bytes);

    Ok(format!(
        "// {} {} (lines {}-{})\n{}",
        format!("{:?}", entry.kind).to_lowercase(),
        entry.name,
        entry.start_line,
        entry.end_line,
        body
    ))
}

fn read_section_mcp(engine: &Engine, path: &Path, line: u32) -> Result<String> {
    let budget = engine.config.total_token_budget;
    let level = engine.config.filter_level;
    section_body(engine, path, line, level, budget)
}

fn text_json(text: impl Into<String>) -> Value {
    json!({"content": [{"type": "text", "text": text.into()}]})
}

fn get_string<'a>(args: &'a Value, key: &str) -> Result<&'a str> {
    args.get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| anyhow::anyhow!("missing {key}"))
}

// Bug 3: accept both `query` (MCP idiom) and `needle` (CLI idiom).
fn get_query(args: &Value) -> Result<&str> {
    args.get("query")
        .and_then(Value::as_str)
        .or_else(|| args.get("needle").and_then(Value::as_str))
        .ok_or_else(|| anyhow::anyhow!("missing query"))
}

fn get_limit(args: &Value, default: usize) -> usize {
    // Accept both `maxResults` (number or integer) and `limit` for parity
    // with the CLI flag.
    args.get("maxResults")
        .and_then(Value::as_u64)
        .or_else(|| {
            args.get("maxResults")
                .and_then(Value::as_f64)
                .map(|v| v.round() as u64)
        })
        .or_else(|| args.get("limit").and_then(Value::as_u64))
        .map(|v| v as usize)
        .filter(|v| *v > 0)
        .unwrap_or(default)
        .max(1)
}

#[derive(Debug, Serialize)]
struct GrepHit {
    path: String,
    line: usize,
    text: String,
}

fn grep_files(root: &Path, query: &str, limit: usize) -> Vec<GrepHit> {
    let query_lower = query.to_lowercase();
    let smart_case = query.chars().any(char::is_uppercase);
    let mut hits = Vec::new();
    for path in super::walk_files(root) {
        if hits.len() >= limit {
            break;
        }
        let Ok(text) = ffs_search::bom::read_file(&path) else {
            continue;
        };
        for (line_idx, line) in text.lines().enumerate() {
            let found = if smart_case {
                line.contains(query)
            } else {
                line.to_lowercase().contains(&query_lower)
            };
            if found {
                hits.push(GrepHit {
                    path: display_path(root, &path),
                    line: line_idx + 1,
                    text: line.trim().to_string(),
                });
                if hits.len() >= limit {
                    break;
                }
            }
        }
    }
    hits
}

/// Multi-pattern OR search (Aho-Corasick) — same semantics as `ffs multi-grep`.
fn multi_grep_files(root: &Path, needles: &[&str], limit: usize) -> Vec<GrepHit> {
    if needles.is_empty() || limit == 0 {
        return Vec::new();
    }
    let case_insensitive = !needles.iter().any(|p| p.chars().any(|c| c.is_uppercase()));
    let Ok(ac) = aho_corasick::AhoCorasickBuilder::new()
        .ascii_case_insensitive(case_insensitive)
        .build(needles)
    else {
        // Fall back to sequential single greps if AC build fails.
        let mut all = Vec::new();
        for n in needles {
            let mut hits = grep_files(root, n, limit.saturating_sub(all.len()));
            all.append(&mut hits);
            if all.len() >= limit {
                break;
            }
        }
        all.truncate(limit);
        return all;
    };

    let mut hits = Vec::new();
    let mut seen_lines: std::collections::HashSet<(String, usize)> =
        std::collections::HashSet::new();

    for path in super::walk_files(root) {
        if hits.len() >= limit {
            break;
        }
        let Ok(text) = ffs_search::bom::read_file(&path) else {
            continue;
        };
        let bytes = text.as_bytes();
        for mat in ac.find_iter(bytes) {
            if hits.len() >= limit {
                break;
            }
            // Map byte offset → 1-based line
            let line = bytes[..mat.start()].iter().filter(|&&b| b == b'\n').count() + 1;
            let key = (display_path(root, &path), line);
            if !seen_lines.insert(key.clone()) {
                continue;
            }
            let line_text = text
                .lines()
                .nth(line.saturating_sub(1))
                .unwrap_or("")
                .trim()
                .to_string();
            hits.push(GrepHit {
                path: key.0,
                line: key.1,
                text: line_text,
            });
        }
    }
    hits
}

// Delegate to the shared core matcher so Windows skips zlob (POSIX-only) and
// path separators are normalized to `/` (regression for #76 / #69).
fn glob_files(root: &Path, pattern: &str, limit: usize) -> Vec<String> {
    ffs_search::glob_matcher::glob_files(root, pattern, limit)
}

#[derive(Debug, Serialize)]
struct CallerHit {
    path: String,
    line: usize,
    text: String,
}

// Coarse caller search: literal-text lookup of `name(` in every file. Mirrors
// the bigram-pre-filter pass the CLI uses but without the on-disk index, which
// keeps the MCP surface self-contained.
fn collect_callers(_state: &mut McpState, root: &Path, name: &str, limit: usize) -> Vec<CallerHit> {
    let mut hits = Vec::new();
    let needle = format!("{name}(");
    for path in super::walk_files(root) {
        if hits.len() >= limit {
            break;
        }
        let Ok(text) = ffs_search::bom::read_file(&path) else {
            continue;
        };
        for (line_idx, line) in text.lines().enumerate() {
            if line.contains(&needle) {
                hits.push(CallerHit {
                    path: display_path(root, &path),
                    line: line_idx + 1,
                    text: line.trim().to_string(),
                });
                if hits.len() >= limit {
                    break;
                }
            }
        }
    }
    hits
}

#[derive(Debug, Serialize)]
struct CalleeHit {
    name: String,
    line: usize,
}

fn collect_callees(state: &mut McpState, _root: &Path, name: &str, limit: usize) -> Vec<CalleeHit> {
    let mut hits = Vec::new();
    let defs = state.engine.handles.symbols.lookup_exact(name);
    for d in defs {
        let Ok(content) = ffs_search::bom::read_file(&d.path) else {
            continue;
        };
        let start = d.line.saturating_sub(1) as usize;
        let end = (d.end_line as usize).min(content.lines().count());
        for (idx, line) in content.lines().enumerate().take(end).skip(start) {
            for word in line.split(|c: char| !c.is_alphanumeric() && c != '_') {
                if word.len() < 3 || word == name {
                    continue;
                }
                if state.engine.handles.symbols.lookup_exact(word).is_empty() {
                    continue;
                }
                hits.push(CalleeHit {
                    name: word.to_string(),
                    line: idx + 1,
                });
                if hits.len() >= limit {
                    return hits;
                }
            }
        }
    }
    hits
}

#[derive(Debug, Serialize)]
struct SiblingHit {
    name: String,
    kind: String,
    path: String,
    line: u32,
}

fn collect_siblings(state: &mut McpState, name: &str, limit: usize) -> Vec<SiblingHit> {
    let defs = state.engine.handles.symbols.lookup_exact(name);
    let mut hits: Vec<SiblingHit> = Vec::new();
    let mut seen: std::collections::HashSet<(String, String, u32)> =
        std::collections::HashSet::new();
    for def in defs {
        let lang = match detect_file_type(&def.path) {
            FileType::Code(l) => l,
            _ => continue,
        };
        let Ok(content) = ffs_search::bom::read_file(&def.path) else {
            continue;
        };
        let mtime = std::fs::metadata(&def.path)
            .and_then(|m| m.modified())
            .unwrap_or(SystemTime::UNIX_EPOCH);
        let outline = state
            .engine
            .handles
            .outlines
            .get_or_compute(&def.path, mtime, &content, lang);
        if let Some((_, peers)) = sibling_peers(&outline, name, def.line) {
            for p in peers {
                let path = def.path.to_string_lossy().to_string();
                let key = (p.name.clone(), path.clone(), p.start_line);
                if !seen.insert(key) {
                    continue;
                }
                hits.push(SiblingHit {
                    name: p.name.clone(),
                    kind: format!("{:?}", p.kind).to_lowercase(),
                    path,
                    line: p.start_line,
                });
                if hits.len() >= limit {
                    return hits;
                }
            }
        }
    }
    hits
}

fn sibling_peers(
    outline: &[OutlineEntry],
    target: &str,
    target_line: u32,
) -> Option<(String, Vec<OutlineEntry>)> {
    if outline
        .iter()
        .any(|e| e.name == target && e.start_line == target_line)
    {
        let peers: Vec<OutlineEntry> = outline
            .iter()
            .filter(|e| !(e.name == target && e.start_line == target_line))
            .cloned()
            .collect();
        return Some(("<file>".to_string(), peers));
    }
    for parent in outline {
        if let Some(found) = sibling_in_children(parent, target, target_line) {
            return Some(found);
        }
    }
    None
}

fn sibling_in_children(
    parent: &OutlineEntry,
    target: &str,
    target_line: u32,
) -> Option<(String, Vec<OutlineEntry>)> {
    if parent
        .children
        .iter()
        .any(|c| c.name == target && c.start_line == target_line)
    {
        let peers: Vec<OutlineEntry> = parent
            .children
            .iter()
            .filter(|c| !(c.name == target && c.start_line == target_line))
            .cloned()
            .collect();
        return Some((parent.name.clone(), peers));
    }
    for c in &parent.children {
        if let Some(found) = sibling_in_children(c, target, target_line) {
            return Some(found);
        }
    }
    None
}

fn read_section_excerpt(path: &Path, start_line: u32, end_line: u32, max_lines: usize) -> String {
    let Ok(content) = ffs_search::bom::read_file(path) else {
        return String::new();
    };
    let start = start_line.saturating_sub(1) as usize;
    let end = (end_line as usize).min(content.lines().count());
    let lines: Vec<&str> = content
        .lines()
        .enumerate()
        .filter(|(i, _)| *i >= start && *i < end)
        .map(|(_, l)| l)
        .take(max_lines)
        .collect();
    lines.join("\n")
}

fn list_imports(path: &Path) -> Vec<String> {
    let lang = match detect_file_type(path) {
        FileType::Code(l) => l,
        _ => return Vec::new(),
    };
    let Ok(content) = ffs_search::bom::read_file(path) else {
        return Vec::new();
    };
    let entries = get_outline_entries(&content, lang);
    entries
        .iter()
        .filter(|e| matches!(e.kind, ffs_symbol::types::OutlineKind::Import))
        .map(|e| e.name.clone())
        .collect()
}

fn render_simple_map(root: &Path, depth: usize) -> Vec<String> {
    let mut counts: std::collections::BTreeMap<PathBuf, usize> = std::collections::BTreeMap::new();
    for path in super::walk_files(root) {
        let rel = path.strip_prefix(root).unwrap_or(&path);
        let mut acc = PathBuf::new();
        for (i, comp) in rel.components().enumerate() {
            if i >= depth {
                break;
            }
            acc.push(comp.as_os_str());
            *counts.entry(acc.clone()).or_default() += 1;
        }
    }
    counts
        .into_iter()
        .map(|(p, n)| format!("{}\t{}", p.display(), n))
        .collect()
}

fn lang_label(lang: &ffs_symbol::types::Lang) -> &'static str {
    use ffs_symbol::types::Lang;
    match lang {
        Lang::Rust => "rust",
        Lang::Python => "python",
        Lang::JavaScript => "javascript",
        Lang::TypeScript => "typescript",
        Lang::Tsx => "tsx",
        Lang::Go => "go",
        Lang::Java => "java",
        Lang::C => "c",
        Lang::Cpp => "cpp",
        Lang::CSharp => "csharp",
        Lang::Ruby => "ruby",
        Lang::Php => "php",
        Lang::Swift => "swift",
        Lang::Kotlin => "kotlin",
        Lang::Scala => "scala",
        Lang::Elixir => "elixir",
        Lang::Verse => "verse",
        Lang::Dockerfile => "dockerfile",
        Lang::Make => "make",
    }
}

fn display_path(root: &Path, path: &Path) -> String {
    path.strip_prefix(root)
        .unwrap_or(path)
        .to_string_lossy()
        .to_string()
}

fn symbol_locs_to_json(hits: Vec<ffs_symbol::symbol_index::SymbolLocation>) -> Value {
    Value::Array(
        hits.into_iter()
            .map(|h| {
                json!({
                    "path": h.path.to_string_lossy(),
                    "line": h.line,
                    "end_line": h.end_line,
                    "kind": h.kind,
                    "weight": h.weight,
                })
            })
            .collect(),
    )
}

fn symbol_glob_to_json(hits: Vec<(String, ffs_symbol::symbol_index::SymbolLocation)>) -> Value {
    Value::Array(
        hits.into_iter()
            .map(|(n, h)| {
                json!({
                    "name": n,
                    "path": h.path.to_string_lossy(),
                    "line": h.line,
                    "end_line": h.end_line,
                    "kind": h.kind,
                    "weight": h.weight,
                })
            })
            .collect(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_workspace_folder_paths_single() {
        let paths = parse_workspace_folder_paths("/Users/me/project");
        assert_eq!(paths, vec![PathBuf::from("/Users/me/project")]);
    }

    #[test]
    fn parse_workspace_folder_paths_windows_multi() {
        let paths = parse_workspace_folder_paths(r"C:\a;C:\b");
        assert_eq!(paths, vec![PathBuf::from(r"C:\a"), PathBuf::from(r"C:\b")]);
    }

    #[test]
    fn parse_workspace_folder_paths_trims_and_skips_empty() {
        let paths = parse_workspace_folder_paths("  \"/a\" ;  ; '/b' \n");
        assert_eq!(paths, vec![PathBuf::from("/a"), PathBuf::from("/b")]);
    }

    #[test]
    fn parse_workspace_folder_paths_keeps_windows_drive() {
        let paths = parse_workspace_folder_paths(r"C:\Users\ADMIN\project");
        assert_eq!(paths, vec![PathBuf::from(r"C:\Users\ADMIN\project")]);
    }

    #[test]
    fn tools_list_includes_object_input_schemas() {
        let td = tempfile::tempdir().unwrap();
        let mut state = McpState::new(Engine::default());
        let result = handle_method(&mut state, td.path(), "tools/list", &Value::Null).unwrap();
        let tools = result["tools"].as_array().unwrap();

        // Bug 2: every tool from the README's "Tools registered" table must
        // be advertised by tools/list. (16 advertised + the MCP-only
        // ffs_glob alias.)
        assert_eq!(tools.len(), 17);
        for tool in tools {
            assert_eq!(tool["inputSchema"]["type"], "object");
            assert!(tool["inputSchema"]["properties"].is_object());
            assert!(tool["inputSchema"]["required"].is_array());
        }
    }

    #[test]
    fn advertised_file_tools_are_callable() {
        let td = tempfile::tempdir().unwrap();
        let root = td.path();
        std::fs::write(root.join("mcp.rs"), "fn mcp_schema() {}\n").unwrap();
        let mut state = McpState::new(Engine::default());

        for (name, args) in [
            ("ffs_find", json!({"query": "mcp.rs"})),
            ("ffs_glob", json!({"pattern": "*.rs"})),
            ("ffs_grep", json!({"query": "mcp_schema"})),
            ("ffs_read", json!({"path": "mcp.rs"})),
        ] {
            let result = handle_tool(&mut state, root, name, &args).unwrap();
            assert!(result["content"][0]["text"].is_string(), "{name}");
        }
    }

    #[test]
    fn find_accepts_needle_alias_for_query() {
        // Bug 3: the CLI uses `<NEEDLE>`, MCP previously required `query`.
        // Now both should work.
        let td = tempfile::tempdir().unwrap();
        let root = td.path();
        std::fs::write(root.join("mcp.rs"), "fn mcp_schema() {}\n").unwrap();
        let mut state = McpState::new(Engine::default());

        let result = handle_tool(&mut state, root, "ffs_find", &json!({"needle": "mcp"})).unwrap();
        assert!(result["content"][0]["text"]
            .as_str()
            .unwrap()
            .contains("mcp.rs"));
    }

    #[test]
    fn missing_tool_errors_clearly() {
        let td = tempfile::tempdir().unwrap();
        let mut state = McpState::new(Engine::default());
        let err = handle_tool(&mut state, td.path(), "ffs_unknown", &Value::Null).unwrap_err();
        assert!(err.to_string().contains("unknown tool"));
    }

    #[test]
    fn initialize_does_not_index_workspace() {
        let td = tempfile::tempdir().unwrap();
        std::fs::write(td.path().join("mcp.rs"), "fn mcp_schema() {}\n").unwrap();
        let mut state = McpState::new(Engine::default());

        let result = handle_method(&mut state, td.path(), "initialize", &Value::Null).unwrap();

        assert_eq!(result["serverInfo"]["name"], "ffs");
        assert!(!state.indexed);
        assert!(state
            .engine
            .handles
            .symbols
            .lookup_exact("mcp_schema")
            .is_empty());
    }

    #[test]
    fn symbol_tool_indexes_lazily() {
        let td = tempfile::tempdir().unwrap();
        std::fs::write(td.path().join("mcp.rs"), "fn mcp_schema() {}\n").unwrap();
        let mut state = McpState::new(Engine::default());

        let result = handle_tool(
            &mut state,
            td.path(),
            "ffs_symbol",
            &json!({"name": "mcp_schema"}),
        )
        .unwrap();

        assert!(state.indexed);
        assert!(result["content"][0]["text"]
            .as_str()
            .unwrap()
            .contains("mcp.rs"));
    }

    #[test]
    fn read_accepts_path_line_span() {
        let td = tempfile::tempdir().unwrap();
        let root = td.path();
        std::fs::write(
            root.join("main.rs"),
            "fn main() {\n    let msg = \"hi\";\n    println!(\"{msg}\");\n}\nfn other() {}\n",
        )
        .unwrap();
        let mut state = McpState::new(Engine::default());

        // `path:line` returns the structural section containing that line.
        let result =
            handle_tool(&mut state, root, "ffs_read", &json!({"path": "main.rs:2"})).unwrap();
        let text = result["content"][0]["text"].as_str().unwrap();
        assert!(text.contains("fn main"), "got: {text}");
        assert!(text.contains("println!"), "got: {text}");
        assert!(!text.contains("fn other"), "got: {text}");

        // No line suffix keeps the whole-file read.
        let whole = handle_tool(&mut state, root, "ffs_read", &json!({"path": "main.rs"})).unwrap();
        let text = whole["content"][0]["text"].as_str().unwrap();
        assert!(text.contains("fn other"), "got: {text}");

        // Absolute path with a line suffix also works.
        let abs = handle_tool(
            &mut state,
            root,
            "ffs_read",
            &json!({"path": format!("{}:1", root.join("main.rs").display())}),
        )
        .unwrap();
        assert!(abs["content"][0]["text"]
            .as_str()
            .unwrap()
            .contains("fn main"));
    }

    #[test]
    fn parse_target_splits_line_suffix() {
        assert_eq!(parse_target("main.rs:7"), ("main.rs", Some(7)));
        assert_eq!(parse_target("src/main.rs:1"), ("src/main.rs", Some(1)));
        assert_eq!(parse_target("main.rs"), ("main.rs", None));
        assert_eq!(parse_target("main.rs:0"), ("main.rs:0", None));
        assert_eq!(parse_target("main.rs:abc"), ("main.rs:abc", None));
        assert_eq!(parse_target("a/b.rs:3:x"), ("a/b.rs:3:x", None));
        assert_eq!(parse_target(r"C:\dir\file.rs"), (r"C:\dir\file.rs", None));
    }
}
