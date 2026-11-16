//! Documents nested too deep: the conversion recurses once per table, group and equation
//! structure, so a document nested past the limit is refused before any of it is converted, and
//! never overflows the stack.

use std::sync::Arc;

use craft_wordmodel as wm;
use wordcraft_doc as doc;

use crate::nesting::{MAX_GROUPS, MAX_MATH, MAX_TABLES};
use crate::{Error, from_shared, to_shared};

/// The stack these tests convert on: small enough that a conversion recursing without bound
/// overflows it long before 5,000 levels, as a thread spawned by an embedder might have.
const SMALL_STACK: usize = 512 * 1024;

/// Far past any limit: unbounded recursion overflows [`SMALL_STACK`] here.
const ABSURD: usize = 5_000;

/// Runs `job` on a thread with [`SMALL_STACK`]; a stack overflow there aborts the test process.
fn on_small_stack<T: Send + 'static>(job: impl FnOnce() -> T + Send + 'static) -> T {
    std::thread::Builder::new().stack_size(SMALL_STACK).spawn(job).unwrap().join().unwrap()
}

/// How a document is nested: `tables` tables one inside another; in the innermost cell a
/// paragraph holding `groups` groups one inside another; inside the innermost group an equation
/// whose structures nest `math` deep.
#[derive(Clone, Copy, Debug)]
struct Nesting {
    tables: usize,
    groups: usize,
    math: usize,
}

impl Nesting {
    fn tables(tables: usize) -> Self {
        Nesting { tables, groups: 0, math: 0 }
    }
    fn groups(groups: usize) -> Self {
        Nesting { tables: 0, groups, math: 0 }
    }
    fn math(math: usize) -> Self {
        Nesting { tables: 0, groups: 0, math }
    }
}

/// Builds a [`Nesting`] in either model, from the inside out, without recursing.
macro_rules! build {
    ($name:ident, $m:ident) => {
        fn $name(nesting: Nesting) -> $m::Document {
            use $m::math::MNode;
            use $m::para::{GroupChild, InlineObject, Paragraph};

            let mut node = MNode::Frac { kind: Default::default(), num: Vec::new(), den: Vec::new() };
            for _ in 1..nesting.math {
                node = MNode::Frac { kind: Default::default(), num: vec![node], den: Vec::new() };
            }
            let math = if nesting.math == 0 { Vec::new() } else { vec![node] };
            let mut object =
                InlineObject::Equation { linear: "x".into(), display: false, math: $m::math::Math { nodes: math, ..Default::default() } };
            for _ in 0..nesting.groups {
                object = InlineObject::Group {
                    w: 10.0,
                    h: 10.0,
                    float: Default::default(),
                    ch_w: 10.0,
                    ch_h: 10.0,
                    children: vec![GroupChild { x: 1.0, y: 2.0, obj: object }],
                    canvas: None,
                };
            }
            let mut paragraph = Paragraph::with_text("\u{fffc}", Default::default());
            paragraph.objects.push(object);
            let mut block = $m::Block::Para(paragraph);
            for _ in 0..nesting.tables {
                let cell = $m::table::Cell { blocks: vec![Arc::new(block)], ..Default::default() };
                let row = $m::table::Row { cells: vec![cell], ..Default::default() };
                block = $m::Block::Table($m::table::Table { rows: vec![row], ..Default::default() });
            }
            let mut document = $m::Document::new();
            document.body = vec![Arc::new(block)];
            document
        }
    };
}

build!(shared, wm);
build!(own, doc);

/// The same nesting in each story that holds blocks, the others left empty.
macro_rules! in_every_story {
    ($name:ident, $m:ident, $build:ident) => {
        fn $name(nesting: Nesting) -> Vec<(&'static str, $m::Document)> {
            let story = || std::mem::take(&mut $build(nesting).body);
            let mut part = $m::Document::new();
            part.parts.insert(3, $m::Part { blocks: story(), ..Default::default() });
            let mut footnote = $m::Document::new();
            footnote.footnote_separators.continuation_notice = Some(story());
            let mut endnote = $m::Document::new();
            endnote.endnote_separators.separator = Some(story());
            let mut glossary = $m::Document::new();
            glossary.glossary.push($m::BuildingBlock { blocks: story(), ..Default::default() });
            vec![
                ("body", $build(nesting)),
                ("part", part),
                ("footnote separators", footnote),
                ("endnote separators", endnote),
                ("glossary", glossary),
            ]
        }
    };
}

in_every_story!(shared_stories, wm, shared);
in_every_story!(own_stories, doc, own);

/// Reads a document nested `nesting` deep, on a small stack; the result is dropped there too.
fn read(nesting: Nesting) -> Vec<(&'static str, Result<(), Error>)> {
    on_small_stack(move || shared_stories(nesting).into_iter().map(|(story, document)| (story, from_shared(document).map(drop))).collect())
}

/// Writes a document nested `nesting` deep, on a small stack; the result is dropped there too.
fn write(nesting: Nesting) -> Vec<(&'static str, Result<(), Error>)> {
    on_small_stack(move || {
        own_stories(nesting)
            .into_iter()
            .map(|(story, document)| {
                let written = to_shared(&document).map(drop);
                // Dropping a document nested this deep recurses as deep as it nests and would
                // overflow here: this measures the bridge, not `Drop`, so the document is leaked
                // (as `from_shared` leaks one it refuses).
                std::mem::forget(document);
                (story, written)
            })
            .collect()
    })
}

fn refused_both_ways(nesting: Nesting) {
    for (story, result) in read(nesting) {
        assert_eq!(result, Err(Error::TooDeep), "reading, {story}: {nesting:?}");
    }
    for (story, result) in write(nesting) {
        assert_eq!(result, Err(Error::TooDeep), "writing, {story}: {nesting:?}");
    }
}

#[test]
fn one_level_past_a_limit_is_refused_both_ways_in_every_story() {
    let at = Nesting { tables: MAX_TABLES, groups: MAX_GROUPS, math: MAX_MATH };
    for nesting in [
        Nesting::tables(MAX_TABLES + 1),
        Nesting::groups(MAX_GROUPS + 1),
        Nesting::math(MAX_MATH + 1),
        Nesting { tables: MAX_TABLES + 1, ..at },
        Nesting { groups: MAX_GROUPS + 1, ..at },
        Nesting { math: MAX_MATH + 1, ..at },
    ] {
        refused_both_ways(nesting);
    }
}

#[test]
fn a_document_nested_absurdly_deep_is_refused_not_overflowing_the_stack() {
    for nesting in [Nesting::tables(ABSURD), Nesting::groups(ABSURD), Nesting::math(ABSURD)] {
        refused_both_ways(nesting);
    }
}

#[test]
fn a_document_at_the_limits_converts_and_round_trips_on_a_small_stack() {
    let at = Nesting { tables: MAX_TABLES, groups: MAX_GROUPS, math: MAX_MATH };
    for nesting in [Nesting::tables(MAX_TABLES), Nesting::groups(MAX_GROUPS), Nesting::math(MAX_MATH), at] {
        for (story, result) in read(nesting).into_iter().chain(write(nesting)) {
            assert_eq!(result, Ok(()), "{story}: {nesting:?}");
        }
        // Both ways round, each starting from its own model; compared here, on the test's own stack.
        let (read_in, back_out, written, back_in) = on_small_stack(move || {
            let read_in = from_shared(shared(nesting)).unwrap();
            let back_out = to_shared(&read_in).unwrap();
            let written = to_shared(&own(nesting)).unwrap();
            let back_in = from_shared(written.clone()).unwrap();
            (read_in, back_out, written, back_in)
        });
        assert_eq!(read_in.body, own(nesting).body, "read: {nesting:?}");
        assert_eq!(back_out.body, shared(nesting).body, "read and written back: {nesting:?}");
        assert_eq!(written.body, shared(nesting).body, "written: {nesting:?}");
        assert_eq!(back_in.body, own(nesting).body, "written and read back: {nesting:?}");
    }
}
