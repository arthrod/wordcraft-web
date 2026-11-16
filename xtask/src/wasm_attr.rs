//! `cargo xtask wasm-attr [--wasm <path>] [--top <N> [--bucket <name>]] [--trait-impls]`: attribute the WASM code section of the
//! `wordcraft-web` build per crate, to decide whether cold-code splitting is worth it.
//!
//! The `wasm-release` profile sets `strip = "symbols"`, which removes the `name` custom
//! section rustc emits — so before analysing, the default path rebuilds with
//! `CARGO_PROFILE_WASM_RELEASE_STRIP=none` (the env override keeps the `name` section;
//! everything else about the build is unchanged). The analysis targets the raw rustc
//! artifact under `target/wasm32-unknown-unknown/wasm-release/`, not the wasm-bindgen
//! output in `dist/` (which renames and post-processes functions).
//!
//! Parsing is done here, without a wasmparser dependency: three sections matter —
//! the import section (id 2) for the function-import count, the code section (id 10)
//! for per-function encoded sizes, and the `name` custom section's function-name
//! subsection for the symbol to attribute each size to. Every read is bounded;
//! malformed input is an error, never a panic.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;

use crate::wasm_budget::read_leb_u64;

/// Where the report goes. `docs/plans/` already holds the implementation plan this
/// measurement informs.
const REPORT_REL: &str = "docs/plans/wasm-attribution.md";

/// Profile env override that keeps the `name` custom section (the profile's
/// `strip = "symbols"` would otherwise remove it and make attribution impossible).
const STRIP_ENV: &str = "CARGO_PROFILE_WASM_RELEASE_STRIP";

/// The bin target of `apps/wordcraft-web` (bins keep hyphens in their artifact name).
const WASM_ARTIFACT: &str = "wordcraft-web.wasm";

#[derive(Debug)]
pub struct Bucket {
    pub name: String,
    pub bytes: usize,
    pub funcs: usize,
}

/// One function body: its encoded size, its primary bucket, and for a trait-impl symbol
/// (`<A as B>::f`) the bucket of the trait's crate when that is a different crate.
#[derive(Debug)]
pub struct Func {
    pub bytes: usize,
    pub bucket: String,
    /// Secondary attribution only: never added to a bucket's primary bytes.
    pub also: Option<String>,
    /// Full demangled name (or a placeholder for a body without a name entry).
    pub name: String,
}

/// Trait-impl cross-attribution for one bucket. Both figures are views of bytes already
/// counted once in the primary table; neither is added to any total.
#[derive(Debug, PartialEq, Eq)]
pub struct TraitImpls {
    pub name: String,
    /// Bytes (and functions) whose primary bucket is this one and that implement another
    /// crate's trait.
    pub foreign_bytes: usize,
    pub foreign_funcs: usize,
    /// Bytes (and functions) primarily in other buckets that implement this crate's traits.
    pub also_bytes: usize,
    pub also_funcs: usize,
}

#[derive(Debug)]
pub struct Attribution {
    pub path: PathBuf,
    pub wasm_bytes: usize,
    /// Payload bytes of the code section (vec count + all function entries).
    pub code_section_bytes: usize,
    /// Sum of the encoded function entries (length prefix + body per function).
    pub function_bytes: usize,
    pub functions: usize,
    /// Sorted by bytes descending, then name.
    pub buckets: Vec<Bucket>,
    /// Every function body, in code-section order.
    pub funcs: Vec<Func>,
}

/// Read a length-prefixed UTF-8 name whose bytes must end within `end`.
fn read_name(bytes: &[u8], pos: &mut usize, end: usize) -> Result<String, String> {
    let len = usize::try_from(read_leb_u64(bytes, pos)?).map_err(|_| "name length too large for usize".to_string())?;
    let name_end = pos.checked_add(len).ok_or_else(|| "name offset overflow".to_string())?;
    if name_end > end {
        return Err("truncated name string".into());
    }
    let slice = bytes.get(*pos..name_end).ok_or_else(|| "name slice out of bounds".to_string())?;
    let name = std::str::from_utf8(slice).map_err(|e| format!("name is not valid UTF-8: {e}"))?;
    *pos = name_end;
    Ok(name.to_string())
}

/// Read a LEB128 that the spec types as u32 (counts, indices, section lengths).
fn read_leb_u32(bytes: &[u8], pos: &mut usize) -> Result<u32, String> {
    u32::try_from(read_leb_u64(bytes, pos)?).map_err(|_| "LEB128 exceeds u32".to_string())
}

/// The code section payload: `<count> <entry>*` where each entry is
/// `<body_size:leb> <body bytes>`. Returns each function's total encoded size
/// (its length prefix plus body), which is exactly what attribution cares about.
fn parse_code_section(bytes: &[u8], start: usize, end: usize) -> Result<Vec<usize>, String> {
    let mut pos = start;
    let count = read_leb_u32(bytes, &mut pos)?;
    let mut sizes = Vec::new();
    for _ in 0..count {
        let before = pos;
        let body = usize::try_from(read_leb_u64(bytes, &mut pos)?).map_err(|_| "function body too large for usize".to_string())?;
        let prefix = pos.saturating_sub(before);
        let body_end = pos.checked_add(body).ok_or_else(|| "function body offset overflow".to_string())?;
        if body_end > end {
            return Err("truncated function body in code section".into());
        }
        pos = body_end;
        sizes.push(prefix.saturating_add(body));
    }
    if pos != end {
        return Err("trailing bytes after the last function body in code section".into());
    }
    Ok(sizes)
}

/// Skip a `limits` descriptor (`<flags> <min> [<max>]`); used by table and memory imports.
fn skip_limits(bytes: &[u8], pos: &mut usize) -> Result<(), String> {
    let flags = *bytes.get(*pos).ok_or_else(|| "truncated limits flags".to_string())?;
    *pos += 1;
    read_leb_u64(bytes, pos)?;
    if flags & 0x01 != 0 {
        read_leb_u64(bytes, pos)?;
    }
    Ok(())
}

/// The import section payload: `<count> <module name> <field name> <kind> <desc>*`.
/// Function indices in the `name` section count imported functions first, so the
/// code section's bodies need this count as an offset into the name map.
fn count_function_imports(bytes: &[u8], start: usize, end: usize) -> Result<u64, String> {
    let mut pos = start;
    let count = read_leb_u64(bytes, &mut pos)?;
    let mut funcs = 0u64;
    for _ in 0..count {
        let _module = read_name(bytes, &mut pos, end)?;
        let _field = read_name(bytes, &mut pos, end)?;
        let kind = *bytes.get(pos).ok_or_else(|| "truncated import kind".to_string())?;
        pos += 1;
        match kind {
            0x00 => {
                funcs = funcs.saturating_add(1);
                read_leb_u64(bytes, &mut pos)?; // type index
            }
            0x01 => {
                pos += 1; // element type
                skip_limits(bytes, &mut pos)?;
            }
            0x02 => skip_limits(bytes, &mut pos)?,
            0x03 => pos = pos.checked_add(2).ok_or_else(|| "import offset overflow".to_string())?,
            0x04 => {
                pos += 1; // tag attribute
                read_leb_u64(bytes, &mut pos)?; // type index
            }
            other => return Err(format!("unknown import kind {other:#04x}")),
        }
        if pos > end {
            return Err("import entry runs past the import section".into());
        }
    }
    if pos != end {
        return Err("trailing bytes in import section".into());
    }
    Ok(funcs)
}

/// Subsection 1 of the `name` custom section: a name-map of `<count> <funcidx, name>*`.
fn parse_name_map(bytes: &[u8], start: usize, end: usize) -> Result<Vec<(u32, String)>, String> {
    let mut pos = start;
    let count = read_leb_u32(bytes, &mut pos)?;
    let mut entries = Vec::new();
    for _ in 0..count {
        let idx = read_leb_u32(bytes, &mut pos)?;
        let name = read_name(bytes, &mut pos, end)?;
        entries.push((idx, name));
    }
    if pos != end {
        return Err("trailing bytes in function-name map".into());
    }
    Ok(entries)
}

/// A custom section (id 0) payload: `<section name> <subsection>*`. For the one named
/// "name", walk the subsections and return subsection 1's function-name map.
fn parse_custom_section(bytes: &[u8], start: usize, end: usize) -> Result<Option<Vec<(u32, String)>>, String> {
    let mut pos = start;
    let section_name = read_name(bytes, &mut pos, end)?;
    if section_name != "name" {
        return Ok(None); // some other custom section (e.g. "target_features"): not ours
    }
    let mut names = None;
    while pos < end {
        let id = *bytes.get(pos).ok_or_else(|| "unexpected EOF in name subsection".to_string())?;
        pos += 1;
        let size = usize::try_from(read_leb_u64(bytes, &mut pos)?).map_err(|_| "subsection too large for usize".to_string())?;
        let sub_end = pos.checked_add(size).ok_or_else(|| "subsection offset overflow".to_string())?;
        if sub_end > end {
            return Err("truncated name subsection".into());
        }
        if id == 1 {
            names = Some(parse_name_map(bytes, pos, sub_end)?);
        }
        pos = sub_end;
    }
    Ok(names)
}

/// The first `::`-separated path segment of a demangled name — the crate. For
/// `<path as trait>::…` forms, the first segment inside the angle brackets; the cut at
/// `:` or `>` keeps nested generics (`<crate::Foo<T> as …>`) from leaking past the crate.
fn first_crate_segment(demangled: &str) -> &str {
    let body = demangled.strip_prefix('<').unwrap_or(demangled);
    let cut = body.find([':', '>']).unwrap_or(body.len());
    body.get(..cut).unwrap_or("")
}

/// A plausible crate name: symbols come from rustc, but hostile or non-Rust names must not
/// create junk buckets.
fn is_identifier(s: &str) -> bool {
    let mut chars = s.chars();
    match chars.next() {
        Some(c) if c.is_ascii_alphabetic() || c == '_' => {}
        _ => return false,
    }
    chars.all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
}

/// Rust primitive type names appear as the first segment of `<str as core::fmt::Display>::fmt`-
/// style symbols; they name a type, not a crate, so they belong in `other`.
const PRIMITIVES: &[&str] = &[
    "bool",
    "char",
    "str",
    "u8",
    "u16",
    "u32",
    "u64",
    "u128",
    "usize",
    "i8",
    "i16",
    "i32",
    "i64",
    "i128",
    "isize",
    "f32",
    "f64",
    "array",
    "slice",
    "tuple",
    "reference",
];

/// Map a demangled first segment to a bucket: workspace crates keep a `wordcraft-*` name,
/// the four support crates group under `rust-std`, a plausible crate name buckets under
/// itself (third-party), anything else (primitives, generics, junk, nothing) is `other`.
fn normalize_bucket(segment: &str) -> String {
    // v0 disambiguators print as `crate[hash]`; strip them.
    let base = segment.split('[').next().unwrap_or(segment);
    if matches!(base, "std" | "core" | "alloc" | "compiler_builtins") {
        return "rust-std".into();
    }
    if base == "wordcraft" || base.starts_with("wordcraft_") {
        return base.replace('_', "-");
    }
    if !PRIMITIVES.contains(&base) && is_identifier(base) {
        return base.to_string();
    }
    "other".into()
}

/// The second `::`-separated path segment of a demangled name, if any, with the same
/// leading-`<` and trailing-generics handling as [`first_crate_segment`] (`<a::b::T<…> as …>`
/// names `b`).
fn second_segment(demangled: &str) -> &str {
    let body = demangled.strip_prefix('<').unwrap_or(demangled);
    let Some(first_cut) = body.find("::") else { return "" };
    let rest = body.get(first_cut + 2..).unwrap_or("");
    let cut = rest.find([':', '>']).unwrap_or(rest.len());
    rest.get(..cut).unwrap_or("")
}

/// Demangle and attribute one symbol. Symbols that demangle to a single word carry no
/// crate path (C shims, link-time thunks), so they land in `other`.
#[cfg(test)]
fn bucket_for_name(mangled: &str) -> String {
    bucket_for_demangled(&rustc_demangle::demangle(mangled).to_string())
}

fn bucket_for_demangled(demangled: &str) -> String {
    if !demangled.contains("::") {
        return "other".into();
    }
    // The one module-level bucket (plan S6): the dialogs are the split-spike's candidate cold
    // region, so they are measured apart from the rest of the UI crate rather than claimed.
    // The first segment carries a v0 disambiguator hash in mangled form
    // (`wordcraft_ui_egui[6c571ced…]::dialogs`), so compare it stripped, as the crate buckets do.
    let first = first_crate_segment(demangled);
    if first.split('[').next().unwrap_or(first) == "wordcraft_ui_egui" && second_segment(demangled) == "dialogs" {
        return "wordcraft-ui-egui::dialogs".into();
    }
    normalize_bucket(first)
}

/// The trait path of a `<A as B>::f` symbol: the text between the ` as ` at angle-bracket
/// depth 1 and the `>` that closes the leading `<`. Nested generics on either side
/// (`<a::T<x::U<V>> as b::Tr<W>>::f`) and `->` in function types do not end it early.
fn trait_path(demangled: &str) -> Option<&str> {
    let body = demangled.strip_prefix('<')?;
    let mut depth = 1usize;
    let mut trait_start = None;
    let mut prev = 0u8;
    for (i, b) in body.bytes().enumerate() {
        match b {
            b'<' => depth = depth.saturating_add(1),
            // `->` is a function-type arrow, not a closing bracket
            b'>' if prev != b'-' => {
                depth = depth.saturating_sub(1);
                if depth == 0 {
                    return body.get(trait_start?..i);
                }
            }
            b' ' if depth == 1 && body.get(i..).is_some_and(|rest| rest.starts_with(" as ")) => trait_start = Some(i.saturating_add(4)),
            _ => {}
        }
        prev = b;
    }
    None
}

/// The bucket of the trait's crate for a trait-impl symbol, when it names a crate other
/// than the one the symbol is primarily attributed to.
fn also_bucket(demangled: &str) -> Option<String> {
    let also = normalize_bucket(first_crate_segment(trait_path(demangled)?));
    (also != "other" && also != normalize_bucket(first_crate_segment(demangled))).then_some(also)
}

/// Classify each function entry by its name in the function-name map; functions without a
/// name entry (or with an unusable one) go to `other`.
/// `imported_funcs` is the number of imported functions: the name map's indices cover
/// imports first, while `sizes` holds only the code section's defined-function bodies,
/// so body `i` is function index `i + imported_funcs`.
pub fn classify(sizes: &[usize], names: &[(u32, String)], imported_funcs: u64) -> Vec<Func> {
    let mut map: BTreeMap<u32, &str> = BTreeMap::new();
    for (idx, name) in names {
        map.insert(*idx, name.as_str());
    }
    let mut funcs = Vec::new();
    for (i, &bytes) in sizes.iter().enumerate() {
        // Function indices come from a u32 count in the code section plus the import
        // count, so try_from cannot fail in practice — but hostile bytes never get to
        // panic on the conversion.
        let idx = u64::try_from(i).ok().and_then(|i| i.checked_add(imported_funcs)).and_then(|i| u32::try_from(i).ok());
        funcs.push(match idx.and_then(|idx| map.get(&idx)) {
            Some(mangled) => {
                let name = rustc_demangle::demangle(mangled).to_string();
                Func { bytes, bucket: bucket_for_demangled(&name), also: also_bucket(&name), name }
            }
            None => Func { bytes, bucket: "other".into(), also: None, name: format!("<unnamed function body {i}>") },
        });
    }
    funcs
}

/// Sum the classified functions per primary bucket: heaviest first, then by name.
pub fn attribute(funcs: &[Func]) -> Vec<Bucket> {
    let mut buckets: BTreeMap<&str, (usize, usize)> = BTreeMap::new();
    for f in funcs {
        let entry = buckets.entry(f.bucket.as_str()).or_insert((0, 0));
        entry.0 = entry.0.saturating_add(f.bytes);
        entry.1 = entry.1.saturating_add(1);
    }
    let mut buckets: Vec<Bucket> = buckets.into_iter().map(|(name, (bytes, funcs))| Bucket { name: name.to_string(), bytes, funcs }).collect();
    buckets.sort_by(|a, b| b.bytes.cmp(&a.bytes).then_with(|| a.name.cmp(&b.name)));
    buckets
}

/// Trait-impl cross-attribution per bucket (see [`TraitImpls`]), largest combined figure
/// first, then by name. Buckets with no trait-impl bytes either way are left out.
pub fn trait_impls(funcs: &[Func]) -> Vec<TraitImpls> {
    fn row<'m>(map: &'m mut BTreeMap<String, TraitImpls>, name: &str) -> &'m mut TraitImpls {
        map.entry(name.to_string()).or_insert_with(|| TraitImpls {
            name: name.to_string(),
            foreign_bytes: 0,
            foreign_funcs: 0,
            also_bytes: 0,
            also_funcs: 0,
        })
    }
    let mut map: BTreeMap<String, TraitImpls> = BTreeMap::new();
    for f in funcs {
        let Some(also) = f.also.as_deref() else { continue };
        let own = row(&mut map, &f.bucket);
        own.foreign_bytes = own.foreign_bytes.saturating_add(f.bytes);
        own.foreign_funcs = own.foreign_funcs.saturating_add(1);
        let other = row(&mut map, also);
        other.also_bytes = other.also_bytes.saturating_add(f.bytes);
        other.also_funcs = other.also_funcs.saturating_add(1);
    }
    let mut rows: Vec<TraitImpls> = map.into_values().collect();
    let weight = |r: &TraitImpls| r.foreign_bytes.saturating_add(r.also_bytes);
    rows.sort_by(|a, b| weight(b).cmp(&weight(a)).then_with(|| a.name.cmp(&b.name)));
    rows
}

/// The `n` largest functions, largest first (ties by name), optionally only those whose
/// primary bucket is `bucket`.
pub fn top_functions<'a>(funcs: &'a [Func], bucket: Option<&str>, n: usize) -> Vec<&'a Func> {
    let mut rows: Vec<&Func> = funcs.iter().filter(|f| bucket.is_none_or(|b| f.bucket == b)).collect();
    rows.sort_by(|a, b| b.bytes.cmp(&a.bytes).then_with(|| a.name.cmp(&b.name)));
    rows.truncate(n);
    rows
}

fn strip_error(path: &Path) -> String {
    format!(
        "the `name` custom section is missing from {}: the binary was stripped (the \
         `wasm-release` profile sets `strip = \"symbols\"`), so function names are gone. \
         Rebuild unstripped with `{STRIP_ENV}=none cargo build -p wordcraft-web \
         --target wasm32-unknown-unknown --profile wasm-release`, or just run \
         `cargo xtask wasm-attr` without `--wasm`, which sets that env var itself.",
        path.display()
    )
}

/// Walk the sections of a wasm binary and attribute its code section. `path` is used in
/// error messages only.
pub fn analyze(path: PathBuf, bytes: &[u8]) -> Result<Attribution, String> {
    if bytes.len() < 8 || bytes.get(..8) != Some(&[0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00]) {
        return Err(format!("{}: not a valid WASM binary", path.display()));
    }
    let mut pos = 8usize;
    let mut code: Option<(usize, usize)> = None;
    let mut imports: Option<(usize, usize)> = None;
    let mut names: Option<Vec<(u32, String)>> = None;
    while pos < bytes.len() {
        let id = *bytes.get(pos).ok_or_else(|| "unexpected EOF reading section id".to_string())?;
        pos += 1;
        let len = usize::try_from(read_leb_u64(bytes, &mut pos)?).map_err(|_| "section too large for usize".to_string())?;
        let end = pos.checked_add(len).ok_or_else(|| "section offset overflow".to_string())?;
        if end > bytes.len() {
            return Err(format!("truncated WASM section id={id}"));
        }
        match id {
            0 => {
                if let Some(parsed) = parse_custom_section(bytes, pos, end)? {
                    names = Some(parsed);
                }
            }
            2 => {
                if imports.is_some() {
                    return Err(format!("{}: duplicate import section", path.display()));
                }
                imports = Some((pos, end));
            }
            10 => {
                if code.is_some() {
                    return Err(format!("{}: duplicate code section", path.display()));
                }
                code = Some((pos, end));
            }
            _ => {}
        }
        pos = end;
    }
    let Some((code_start, code_end)) = code else {
        return Err(format!("{}: no code section (id 10) found", path.display()));
    };
    let Some(names) = names else {
        return Err(strip_error(&path));
    };
    let imported_funcs = match imports {
        Some((start, end)) => count_function_imports(bytes, start, end)?,
        None => 0,
    };
    let sizes = parse_code_section(bytes, code_start, code_end)?;
    let functions = sizes.len();
    let function_bytes = sizes.iter().fold(0usize, |acc, s| acc.saturating_add(*s));
    let funcs = classify(&sizes, &names, imported_funcs);
    let buckets = attribute(&funcs);
    Ok(Attribution {
        path,
        wasm_bytes: bytes.len(),
        code_section_bytes: code_end.saturating_sub(code_start),
        function_bytes,
        functions,
        buckets,
        funcs,
    })
}

/// Bytes in buckets matching `pred` (used for the plan-focused group summary).
fn group_bytes(a: &Attribution, pred: impl Fn(&Bucket) -> bool) -> usize {
    a.buckets.iter().filter(|b| pred(b)).fold(0usize, |acc, b| acc.saturating_add(b.bytes))
}

fn pct(part: usize, whole: usize) -> String {
    if whole == 0 {
        return "0.0%".into();
    }
    format!("{:.1}%", part as f64 / whole as f64 * 100.0)
}

/// The plan cares about three groups: our crates (cold-code splitting targets), rust-std
/// (only linker GC moves it), and third-party crates (only dependency/feature cuts move them).
fn group_note(a: &Attribution) -> Vec<String> {
    let ours = group_bytes(a, |b| b.name.starts_with("wordcraft-"));
    let std = group_bytes(a, |b| b.name == "rust-std");
    let other = group_bytes(a, |b| b.name == "other");
    let third_party = a.function_bytes.saturating_sub(ours).saturating_sub(std).saturating_sub(other);
    vec![
        format!("- `wordcraft-*` (ours to split or slim): {} bytes ({})", ours, pct(ours, a.function_bytes)),
        format!("- `rust-std` (shrinks via linker GC / panic paths only): {} bytes ({})", std, pct(std, a.function_bytes)),
        format!("- third-party crates (shrink via dependency and feature cuts): {} bytes ({})", third_party, pct(third_party, a.function_bytes)),
        format!("- `other` (unnamed or foreign symbols): {} bytes ({})", other, pct(other, a.function_bytes)),
    ]
}

fn render_text(a: &Attribution) -> String {
    let mut out = format!("WordCraft Web WASM code attribution ({}):\n", a.path.display());
    out.push_str(&format!("  wasm total:    {} bytes\n", a.wasm_bytes));
    out.push_str(&format!(
        "  code section:  {} bytes across {} functions (entries: {} bytes)\n\n",
        a.code_section_bytes, a.functions, a.function_bytes
    ));
    out.push_str(&format!("{:<34} {:>12} {:>9} {:>8}\n", "bucket", "code bytes", "% code", "funcs"));
    for b in &a.buckets {
        out.push_str(&format!("{:<34} {:>12} {:>9} {:>8}\n", b.name, b.bytes, pct(b.bytes, a.function_bytes), b.funcs));
    }
    out.push_str(&format!("{:<34} {:>12} {:>9} {:>8}\n", "TOTAL", a.function_bytes, pct(a.function_bytes, a.function_bytes), a.functions));
    out
}

/// The `--top` view: one row per function, full demangled name last so nothing is cut.
fn render_top(a: &Attribution, n: usize, bucket: Option<&str>) -> String {
    let scope = bucket.map_or_else(|| "all buckets".to_string(), |b| format!("`{b}`"));
    let (count, bytes) = a
        .funcs
        .iter()
        .filter(|f| bucket.is_none_or(|b| f.bucket == b))
        .fold((0usize, 0usize), |(c, s), f| (c.saturating_add(1), s.saturating_add(f.bytes)));
    let rows = top_functions(&a.funcs, bucket, n);
    let mut out = format!("top {} of {count} functions in {scope} ({bytes} bytes), largest first:\n", rows.len());
    out.push_str(&format!("{:>10}  {:<28} {:<20} name\n", "bytes", "bucket", "also (trait crate)"));
    for f in rows {
        out.push_str(&format!("{:>10}  {:<28} {:<20} {}\n", f.bytes, f.bucket, f.also.as_deref().unwrap_or("-"), f.name));
    }
    out
}

/// The `--trait-impls` view: per bucket, the primary bytes next to both trait-impl figures.
fn render_trait_impls(a: &Attribution) -> String {
    let mut out = String::from(
        "trait impls (`<A as B>::f`) across crates. Primary bytes count each function once, under A's crate.\n\
         `impls of foreign traits` is the part of that primary figure whose trait B is another crate's;\n\
         `also attributed` is code counted under other buckets that implements this crate's traits.\n\
         Neither is added to any total.\n\n",
    );
    out.push_str(&format!(
        "{:<34} {:>12} {:>24} {:>7} {:>16} {:>7}\n",
        "bucket", "primary", "impls of foreign traits", "funcs", "also attributed", "funcs"
    ));
    for r in trait_impls(&a.funcs) {
        let primary = a.buckets.iter().find(|b| b.name == r.name).map_or(0, |b| b.bytes);
        out.push_str(&format!(
            "{:<34} {:>12} {:>24} {:>7} {:>16} {:>7}\n",
            r.name, primary, r.foreign_bytes, r.foreign_funcs, r.also_bytes, r.also_funcs
        ));
    }
    out
}

fn render_markdown(a: &Attribution, date: &str) -> String {
    let mut out = String::new();
    out.push_str("# WASM code-section attribution (`wordcraft-web`)\n\n");
    out.push_str(&format!("- wasm: `{}` ({} bytes)\n", a.path.display(), a.wasm_bytes));
    out.push_str(&format!("- date: {date}\n"));
    out.push_str(&format!(
        "- code section (id 10): {} bytes across {} function bodies ({} bytes of entries)\n",
        a.code_section_bytes, a.functions, a.function_bytes
    ));
    out.push_str(
        "- method: each function's encoded size (length prefix + body) from the code section, \
         attributed by crate via the `name` custom section's function names, demangled \
         with `rustc-demangle`; functions without a name entry (LTO internals) fall to `other`. \
         Produced by `cargo xtask wasm-attr`.\n\n",
    );
    out.push_str("| bucket | code bytes | % of code | functions |\n");
    out.push_str("| --- | ---: | ---: | ---: |\n");
    for b in &a.buckets {
        out.push_str(&format!("| `{}` | {} | {} | {} |\n", b.name, b.bytes, pct(b.bytes, a.function_bytes), b.funcs));
    }
    out.push_str(&format!("| **TOTAL** | {} | {} | {} |\n\n", a.function_bytes, pct(a.function_bytes, a.function_bytes), a.functions));
    out.push_str("## What the plan cares about\n\n");
    out.push_str(&group_note(a).join("\n"));
    out.push('\n');
    out
}

/// Parsed arguments. `top` and `trait_impls` are queries: they print their view and leave
/// the report file alone.
#[derive(Debug, Default, PartialEq, Eq)]
struct Opts {
    wasm: Option<PathBuf>,
    top: Option<usize>,
    bucket: Option<String>,
    trait_impls: bool,
}

/// `--wasm <path>`, `--top <N> [--bucket <name>]`, `--trait-impls`; anything else is a
/// usage error.
fn parse_args(args: &[&str]) -> Result<Opts, String> {
    const USAGE: &str = "usage: cargo xtask wasm-attr [--wasm <path>] [--top <N> [--bucket <name>]] [--trait-impls]";
    let mut opts = Opts::default();
    let mut i = 0usize;
    while let Some(arg) = args.get(i) {
        match *arg {
            "--wasm" => {
                let value = args.get(i + 1).ok_or_else(|| format!("--wasm needs a path\n{USAGE}"))?;
                opts.wasm = Some(PathBuf::from(value));
                i += 2;
            }
            "--top" => {
                let value = args.get(i + 1).ok_or_else(|| format!("--top needs a count\n{USAGE}"))?;
                let n =
                    value.parse::<usize>().ok().filter(|n| *n > 0).ok_or_else(|| format!("--top needs a positive count, got `{value}`\n{USAGE}"))?;
                opts.top = Some(n);
                i += 2;
            }
            "--bucket" => {
                let value = args.get(i + 1).ok_or_else(|| format!("--bucket needs a bucket name\n{USAGE}"))?;
                opts.bucket = Some((*value).to_string());
                i += 2;
            }
            "--trait-impls" => {
                opts.trait_impls = true;
                i += 1;
            }
            other => return Err(format!("unknown argument `{other}`\n{USAGE}")),
        }
    }
    if opts.bucket.is_some() && opts.top.is_none() {
        return Err(format!("--bucket filters the --top listing and needs it\n{USAGE}"));
    }
    Ok(opts)
}

/// Rebuild the wasm unstripped (the env var keeps the `name` section) and return the
/// artifact path under the target dir cargo reports.
fn build_and_find() -> Result<PathBuf, String> {
    let mut cmd = crate::cargo();
    cmd.args(["build", "-p", "wordcraft-web", "--target", "wasm32-unknown-unknown", "--profile", "wasm-release"]).env(STRIP_ENV, "none");
    eprintln!("$ cargo build -p wordcraft-web --target wasm32-unknown-unknown --profile wasm-release ({STRIP_ENV}=none)");
    let status = cmd.status().map_err(|e| format!("cargo build failed to spawn: {e}"))?;
    if !status.success() {
        return Err(format!("cargo build exited with {status}"));
    }
    let meta = crate::metadata()?;
    let target_dir = meta.get("target_directory").and_then(|v| v.as_str()).ok_or_else(|| "cargo metadata: no target_directory".to_string())?;
    let wasm = Path::new(target_dir).join("wasm32-unknown-unknown/wasm-release").join(WASM_ARTIFACT);
    if !wasm.is_file() {
        return Err(format!("expected {} after the build", wasm.display()));
    }
    Ok(wasm)
}

/// UTC date for the report header; `date` is ubiquitous where cargo runs, and a missing
/// it must not fail the analysis (a placeholder is enough — no chrono dependency here).
fn utc_date() -> String {
    Command::new("date")
        .args(["-u", "+%Y-%m-%d"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| "unknown (`date` unavailable)".into())
}

pub fn run(root: &Path, args: &[&str]) -> Result<(), String> {
    let opts = parse_args(args)?;
    let wasm_path = match opts.wasm {
        Some(p) => p,
        None => build_and_find()?,
    };
    let bytes = std::fs::read(&wasm_path).map_err(|e| format!("{}: {e}", wasm_path.display()))?;
    let attribution = analyze(wasm_path, &bytes)?;
    if opts.top.is_some() || opts.trait_impls {
        if opts.trait_impls {
            print!("{}", render_trait_impls(&attribution));
        }
        if let Some(n) = opts.top {
            print!("{}", render_top(&attribution, n, opts.bucket.as_deref()));
        }
        return Ok(());
    }
    print!("{}", render_text(&attribution));
    let report = root.join(REPORT_REL);
    if let Some(parent) = report.parent() {
        std::fs::create_dir_all(parent).map_err(|e| format!("create {}: {e}", parent.display()))?;
    }
    std::fs::write(&report, render_markdown(&attribution, &utc_date())).map_err(|e| format!("{}: {e}", report.display()))?;
    println!("\nwrote {}", report.display());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn leb(n: u64) -> Vec<u8> {
        let mut out = Vec::new();
        let mut n = n;
        loop {
            let mut b = (n & 0x7f) as u8;
            n >>= 7;
            if n != 0 {
                b |= 0x80;
            }
            out.push(b);
            if n == 0 {
                break;
            }
        }
        out
    }

    fn section(id: u8, payload: &[u8]) -> Vec<u8> {
        let mut out = vec![id];
        out.extend(leb(payload.len() as u64));
        out.extend_from_slice(payload);
        out
    }

    fn wasm_bytes(sections: &[Vec<u8>]) -> Vec<u8> {
        let mut out = vec![0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00];
        for s in sections {
            out.extend_from_slice(s);
        }
        out
    }

    /// A code section payload with one body of `len` padding bytes per function. Entry
    /// sizes are therefore `len` + 1 (the single-byte size prefix).
    fn code_payload(body_lens: &[usize]) -> Vec<u8> {
        let mut p = leb(body_lens.len() as u64);
        for &len in body_lens {
            p.extend(leb(len as u64));
            p.extend(std::iter::repeat_n(0u8, len));
        }
        p
    }

    /// The `name` custom section with subsection 1 (function names) mapping `entries`.
    fn name_section(entries: &[(u32, &str)]) -> Vec<u8> {
        let mut sub = leb(entries.len() as u64);
        for &(idx, name) in entries {
            sub.extend(leb(u64::from(idx)));
            sub.extend(leb(name.len() as u64));
            sub.extend_from_slice(name.as_bytes());
        }
        let mut payload = leb(4);
        payload.extend_from_slice(b"name");
        payload.push(1);
        payload.extend(leb(sub.len() as u64));
        payload.extend(sub);
        section(0, &payload)
    }

    /// An import section with one function import (`env`.`imp`, kind 0x00, typeidx 0)
    /// and one memory import (`env`.`mem`, kind 0x02, limits flags 0 min 1).
    fn import_section() -> Vec<u8> {
        let mut p = leb(2);
        for (field, kind, desc) in [("imp", 0x00u8, vec![0x00]), ("mem", 0x02, vec![0x00, 0x01])] {
            p.extend(leb(3));
            p.extend_from_slice(b"env");
            p.extend(leb(field.len() as u64));
            p.extend_from_slice(field.as_bytes());
            p.push(kind);
            p.extend(desc);
        }
        section(2, &p)
    }

    fn sample(names: &[(u32, &str)]) -> Vec<u8> {
        wasm_bytes(&[section(10, &code_payload(&[10, 20, 5])), name_section(names)])
    }

    #[test]
    fn attributes_functions_to_crates_and_totals_match() {
        let bytes = sample(&[(0, "wordcraft_engine::x::f"), (1, "core::fmt::g"), (2, "egui::ui::h")]);
        let a = analyze(PathBuf::from("t.wasm"), &bytes).unwrap();
        assert_eq!(a.functions, 3);
        // entries: 10+1, 20+1, 5+1 prefix bytes
        assert_eq!(a.function_bytes, 11 + 21 + 6);
        let bucket = |n: &str| a.buckets.iter().find(|b| b.name == n);
        assert_eq!(bucket("wordcraft-engine").map(|b| (b.bytes, b.funcs)), Some((11, 1)));
        assert_eq!(bucket("rust-std").map(|b| (b.bytes, b.funcs)), Some((21, 1)));
        assert_eq!(bucket("egui").map(|b| (b.bytes, b.funcs)), Some((6, 1)));
        // every function lands somewhere, and the buckets are heaviest first
        let total: usize = a.buckets.iter().map(|b| b.bytes).sum();
        assert_eq!(total, a.function_bytes);
        assert!(a.buckets.windows(2).all(|w| w[0].bytes >= w[1].bytes));
    }

    /// S6: the dialogs get their own module-level bucket, so "the dialogs are hot/cold" is a
    /// measured row instead of a claim buried in the crate's total. Only that module splits out;
    /// every other `wordcraft_ui_egui` module stays in the crate bucket.
    #[test]
    fn ui_egui_dialogs_get_their_own_bucket() {
        // Five entries (sizes 10, 20, 5, 7, 7 + a length prefix each): three dialog forms —
        // plain, trait-impl, and the v0-mangled shape rustc emits in the name section — and
        // two non-dialog forms to prove the crate bucket keeps them.
        let bytes = wasm_bytes(&[
            section(10, &code_payload(&[10, 20, 5, 7, 7])),
            name_section(&[
                (0, "wordcraft_ui_egui::dialogs::text_dialogs::show"),
                (1, "<wordcraft_ui_egui::dialogs::para::D as core::fmt::Debug>::fmt"),
                (2, "wordcraft_ui_egui::ribbon::draw"),
                (3, "wordcraft_ui_egui[6c571ced3128981]::dialogs::dispatch::body"),
                (4, "wordcraft_ui_egui[6c571ced3128981]::ribbon::x"),
            ]),
        ]);
        let a = analyze(PathBuf::from("t.wasm"), &bytes).unwrap();
        let bucket = |n: &str| a.buckets.iter().find(|b| b.name == n);
        assert_eq!(bucket("wordcraft-ui-egui::dialogs").map(|b| (b.funcs, b.bytes)), Some((3, 11 + 21 + 8)), "{:?}", a.buckets);
        assert_eq!(bucket("wordcraft-ui-egui").map(|b| (b.funcs, b.bytes)), Some((2, 6 + 8)), "{:?}", a.buckets);
    }

    #[test]
    fn names_that_are_already_demangled_and_mangled_forms_bucket_the_same() {
        let bytes = sample(&[
            (0, "_ZN16wordcraft_engine1x1f17h0123456789abcdefE"),
            (1, "<wordcraft_engine::Foo<core::cell::Cell<u8>> as core::fmt::Debug>::fmt"),
        ]);
        let a = analyze(PathBuf::from("t.wasm"), &bytes).unwrap();
        // 10+1 and 20+1 both land in wordcraft-engine, regardless of mangling form; the
        // unnamed third function falls to `other`.
        assert_eq!(a.buckets.first().map(|b| (b.name.as_str(), b.bytes, b.funcs)), Some(("wordcraft-engine", 32, 2)));
        assert_eq!(a.buckets.get(1).map(|b| (b.name.as_str(), b.bytes)), Some(("other", 6)));
    }

    #[test]
    fn unnamed_unusable_and_foreign_names_go_to_other() {
        let bytes = sample(&[(0, "malloc"), (1, "not a crate name::leading spaces are not identifiers")]);
        let a = analyze(PathBuf::from("t.wasm"), &bytes).unwrap();
        // function 2 has no name entry at all; the two named ones carry no usable crate
        assert_eq!(a.buckets.first().map(|b| (b.name.as_str(), b.bytes, b.funcs)), Some(("other", 11 + 21 + 6, 3)));
    }

    #[test]
    fn v0_disambiguators_are_stripped_from_crate_names() {
        assert_eq!(bucket_for_name("wordcraft_engine[1f2a3b4c]::layout::break_lines"), "wordcraft-engine");
        assert_eq!(bucket_for_name("compiler_builtins::mem::memcpy"), "rust-std");
        assert_eq!(bucket_for_name("alloc::vec::Vec<T>::push"), "rust-std");
        assert_eq!(bucket_for_name("serde::ser::serialize"), "serde");
        assert_eq!(bucket_for_name("js_sys::Reflect::get"), "js_sys");
        assert_eq!(bucket_for_name("<str as core::fmt::Display>::fmt"), "other");
        assert_eq!(bucket_for_name("<usize as core::ops::Add>::add"), "other");
        assert_eq!(bucket_for_name(""), "other");
        assert_eq!(bucket_for_name("__rust_alloc"), "other");
    }

    /// Imported functions occupy the first function indices, so the name map's indices
    /// are offset by the import count relative to the code section's body order.
    #[test]
    fn imported_functions_offset_the_name_map_indices() {
        // One function import takes index 0; the three defined bodies are indices 1-3.
        let bytes = wasm_bytes(&[
            import_section(),
            section(10, &code_payload(&[10, 20, 5])),
            name_section(&[(0, "wordcraft_doc::imported"), (1, "wordcraft_engine::x::f"), (2, "core::fmt::g"), (3, "egui::ui::h")]),
        ]);
        let a = analyze(PathBuf::from("t.wasm"), &bytes).unwrap();
        assert_eq!(a.functions, 3);
        assert_eq!(a.function_bytes, 11 + 21 + 6);
        let bucket = |n: &str| a.buckets.iter().find(|b| b.name == n);
        assert_eq!(bucket("wordcraft-engine").map(|b| (b.bytes, b.funcs)), Some((11, 1)));
        assert_eq!(bucket("rust-std").map(|b| (b.bytes, b.funcs)), Some((21, 1)));
        assert_eq!(bucket("egui").map(|b| (b.bytes, b.funcs)), Some((6, 1)));
        // the import's own name is never attributed to a body
        assert!(a.buckets.iter().all(|b| b.funcs == 1), "{:?}", a.buckets);
    }

    #[test]
    fn missing_or_stripped_name_section_is_a_clear_error() {
        let bytes = wasm_bytes(&[section(10, &code_payload(&[8]))]);
        let err = analyze(PathBuf::from("t.wasm"), &bytes).unwrap_err();
        assert!(err.contains("stripped"), "{err}");
        assert!(err.contains("CARGO_PROFILE_WASM_RELEASE_STRIP=none"), "{err}");
    }

    #[test]
    fn malformed_and_truncated_inputs_error_rather_than_panic() {
        let cases: Vec<Vec<u8>> = vec![
            Vec::new(),
            b"\0asX\x01\0\0\0".to_vec(),
            // section length beyond EOF
            wasm_bytes(&[vec![10, 0xff, 0x01]]),
            // code section: count says 2, data for 1
            wasm_bytes(&[section(10, &[2, 4, 0, 0, 0, 0])]),
            // code section: trailing bytes after the declared functions
            wasm_bytes(&[section(10, &[1, 1, 0, 0])]),
            // name string runs past the end of the name section
            wasm_bytes(&[section(10, &code_payload(&[1])), section(0, &[4, b'n', b'a', b'm', b'e', 1, 5, 1, 99, 5, b'f', b'u'])]),
            // name-map count says 2 but only one entry is present
            wasm_bytes(&[section(10, &code_payload(&[1])), section(0, &[4, b'n', b'a', b'm', b'e', 1, 5, 2, 0, 3, b'a', b':', b':', b'b'])]),
            // unterminated LEB128 runs into EOF
            wasm_bytes(&[vec![10, 0x80, 0x80, 0x80]]),
            // duplicate code sections
            wasm_bytes(&[section(10, &code_payload(&[1])), section(10, &code_payload(&[1])), name_section(&[(0, "a::b")])]),
            // non-UTF-8 function name
            wasm_bytes(&[section(10, &code_payload(&[1])), section(0, &[4, b'n', b'a', b'm', b'e', 1, 7, 1, 2, 0xff, 0xfe, 0x01])]),
        ];
        for (i, case) in cases.iter().enumerate() {
            let res = analyze(PathBuf::from(format!("case{i}.wasm")), case);
            assert!(res.is_err(), "case {i} should error: {case:02x?} -> {res:?}");
        }
    }

    #[test]
    fn missing_code_section_is_an_error() {
        let bytes = wasm_bytes(&[name_section(&[(0, "a::b")])]);
        let err = analyze(PathBuf::from("t.wasm"), &bytes).unwrap_err();
        assert!(err.contains("no code section"), "{err}");
    }

    #[test]
    fn zero_function_code_section_attributes_nothing() {
        let bytes = wasm_bytes(&[section(10, &code_payload(&[])), name_section(&[])]);
        let a = analyze(PathBuf::from("t.wasm"), &bytes).unwrap();
        assert_eq!(a.functions, 0);
        assert_eq!(a.function_bytes, 0);
        assert!(a.buckets.is_empty());
        assert!(render_text(&a).contains("TOTAL"));
        assert!(render_markdown(&a, "2025-01-01").contains("**TOTAL**"));
    }

    #[test]
    fn empty_function_still_counts_with_its_prefix() {
        let bytes = wasm_bytes(&[section(10, &code_payload(&[0, 3])), name_section(&[(0, "std::f"), (1, "wordcraft_doc::g")])]);
        let a = analyze(PathBuf::from("t.wasm"), &bytes).unwrap();
        assert_eq!(a.function_bytes, 1 + 4);
        assert_eq!(a.buckets.iter().map(|b| b.bytes).collect::<Vec<_>>(), vec![4, 1]);
    }

    #[test]
    fn parse_args_takes_exactly_one_wasm_path() {
        assert!(parse_args(&[]).unwrap().wasm.is_none());
        assert_eq!(parse_args(&["--wasm", "/tmp/a.wasm"]).unwrap().wasm, Some(PathBuf::from("/tmp/a.wasm")));
        assert!(parse_args(&["--wasm"]).is_err());
        assert!(parse_args(&["--wasm", "a", "extra"]).is_err());
    }

    #[test]
    fn parse_args_takes_top_bucket_and_trait_impls() {
        assert_eq!(parse_args(&[]).unwrap(), Opts::default());
        let o = parse_args(&["--top", "50", "--bucket", "web_sys", "--wasm", "a.wasm", "--trait-impls"]).unwrap();
        assert_eq!(o, Opts { wasm: Some(PathBuf::from("a.wasm")), top: Some(50), bucket: Some("web_sys".into()), trait_impls: true });
        assert!(parse_args(&["--top"]).is_err());
        assert!(parse_args(&["--top", "many"]).is_err());
        assert!(parse_args(&["--top", "-1"]).is_err());
        assert!(parse_args(&["--top", "0"]).is_err());
        assert!(parse_args(&["--bucket"]).is_err());
        // a bucket filter without a listing to filter is a mistake, not a silent no-op
        assert!(parse_args(&["--bucket", "web_sys"]).is_err());
    }

    #[test]
    fn trait_path_survives_nested_generics_and_fn_arrows() {
        assert_eq!(trait_path("<web_sys::T as wasm_bindgen::convert::FromWasmAbi>::from_abi"), Some("wasm_bindgen::convert::FromWasmAbi"));
        assert_eq!(trait_path("<a::T<x::U<V>> as b::Tr<W<X>>>::f::{{closure}}"), Some("b::Tr<W<X>>"));
        assert_eq!(trait_path("<dyn core::ops::Fn(A) -> R<S> as b::Tr>::f"), Some("b::Tr"));
        // the inner `as` belongs to the self type, the outer one names the trait
        assert_eq!(trait_path("<<a::T as b::Iter>::Item as c::Tr>::f"), Some("c::Tr"));
        assert_eq!(trait_path("<a::T>::f"), None);
        assert_eq!(trait_path("a::b::f"), None);
        assert_eq!(trait_path("<a::T as b::Tr"), None);
        assert_eq!(trait_path("<"), None);
        assert_eq!(trait_path(""), None);
    }

    /// The V3 question: a `<A as B>::f` body stays wholly in A's bucket for the primary table,
    /// and is reported a second time under B's crate without entering any total.
    #[test]
    fn trait_impls_are_counted_under_both_crates_without_double_counting() {
        let bytes = wasm_bytes(&[
            section(10, &code_payload(&[10, 20, 5, 7, 3])),
            name_section(&[
                (0, "<web_sys::T as wasm_bindgen::convert::FromWasmAbi>::from_abi"),
                (1, "<web_sys::Foo<core::cell::Cell<u8>> as serde::Serialize>::serialize::{{closure}}"),
                (2, "web_sys::plain::f"),
                // an impl of the crate's own trait is not cross-crate
                (3, "<web_sys::T as web_sys::Tr>::g"),
                (4, "<usize as wasm_bindgen::convert::IntoWasmAbi>::into_abi"),
            ]),
        ]);
        let a = analyze(PathBuf::from("t.wasm"), &bytes).unwrap();
        // primary table: exactly as before, every byte counted once
        let bucket = |n: &str| a.buckets.iter().find(|b| b.name == n).map(|b| (b.bytes, b.funcs));
        assert_eq!(bucket("web_sys"), Some((11 + 21 + 6 + 8, 4)));
        assert_eq!(bucket("other"), Some((4, 1)));
        assert_eq!(bucket("wasm_bindgen"), None);
        assert_eq!(bucket("serde"), None);
        assert_eq!(a.buckets.iter().map(|b| b.bytes).sum::<usize>(), a.function_bytes);
        // secondary figures
        let row = |name: &str, foreign: (usize, usize), also: (usize, usize)| TraitImpls {
            name: name.into(),
            foreign_bytes: foreign.0,
            foreign_funcs: foreign.1,
            also_bytes: also.0,
            also_funcs: also.1,
        };
        assert_eq!(
            trait_impls(&a.funcs),
            vec![row("web_sys", (32, 2), (0, 0)), row("serde", (0, 0), (21, 1)), row("wasm_bindgen", (0, 0), (15, 2)), row("other", (4, 1), (0, 0))]
        );
        let text = render_trait_impls(&a);
        assert!(text.lines().any(|l| l.split_whitespace().collect::<Vec<_>>() == ["web_sys", "46", "32", "2", "0", "0"]), "{text}");
        assert!(text.lines().any(|l| l.split_whitespace().collect::<Vec<_>>() == ["serde", "0", "0", "0", "21", "1"]), "{text}");
        // the default table is unaffected
        assert!(!render_text(&a).contains("serde"));
    }

    #[test]
    fn top_functions_lists_the_largest_first_with_full_names() {
        let bytes = wasm_bytes(&[
            section(10, &code_payload(&[10, 20, 5, 20, 30])),
            name_section(&[
                (0, "web_sys::small"),
                (1, "<web_sys::T as wasm_bindgen::closure::WasmClosure>::describe::{{closure}}"),
                (2, "_ZN6js_sys7Reflect3get17h0123456789abcdefE"),
                (3, "web_sys::a_tie"),
            ]),
        ]);
        let a = analyze(PathBuf::from("t.wasm"), &bytes).unwrap();
        let rows = |bucket: Option<&str>, n: usize| top_functions(&a.funcs, bucket, n).iter().map(|f| (f.bytes, f.name.as_str())).collect::<Vec<_>>();
        assert_eq!(
            rows(None, 3),
            vec![
                (31, "<unnamed function body 4>"),
                (21, "<web_sys::T as wasm_bindgen::closure::WasmClosure>::describe::{{closure}}"),
                (21, "web_sys::a_tie")
            ]
        );
        assert_eq!(rows(Some("web_sys"), 2).len(), 2);
        assert_eq!(rows(Some("web_sys"), 99).iter().map(|r| r.0).collect::<Vec<_>>(), vec![21, 21, 11]);
        assert_eq!(rows(Some("js_sys"), 5), vec![(6, "js_sys::Reflect::get::h0123456789abcdef")]);
        assert!(rows(Some("no_such_bucket"), 5).is_empty());
        let text = render_top(&a, 2, Some("web_sys"));
        let body: Vec<&str> = text.lines().skip(2).collect();
        assert_eq!(body.len(), 2, "{text}");
        assert!(body[0].ends_with("<web_sys::T as wasm_bindgen::closure::WasmClosure>::describe::{{closure}}"), "{text}");
        assert!(body[0].contains("wasm_bindgen "), "the secondary bucket is shown: {text}");
        assert!(text.starts_with("top 2 of 3 functions in `web_sys` (53 bytes)"), "{text}");
    }
}
