//! How deep a document nests, measured before it is converted.
//!
//! The conversion recurses once for each table in a table's cell, each group in a group and each
//! equation structure in another, so a document nested deep enough would overflow the stack: a
//! crash, not an error. (The JSON bridge this crate replaced was bounded by `serde_json`'s
//! recursion limit.) Each kind of nesting has its own limit, because a level of each costs the
//! conversion a very different amount of stack. The walk that measures recurses too, but it stops
//! descending the moment a limit is passed, so its own depth is bounded by the limits.
//!
//! The limits are above what the shared readers make (craft-docx reads tables 24 deep;
//! craft-doc-binary groups 12 deep, craft-docx 8; craft-math equations 64 structures deep), and a
//! document at all three limits at once converts on a 512 KiB stack (`tests/depth.rs`).
//!
//! Measured 2026-11-16 on a thread with a 512 KiB stack, the deepest that converted and dropped
//! without overflowing, reading / writing. With the workspace's dev and test profile (`opt-level`
//! 1 for this crate, 2 for its dependencies), each kind alone: tables 92 / 151, groups 276 / 536,
//! equation structures 898 / 1,423; tables with groups and structures at their limits: 79 / 139,
//! so the table limit leaves more than half of the reading headroom unused. With everything at
//! `opt-level` 0, each alone: tables 36 / 86, groups 53 / 133, structures 123 / 261; tables with
//! the other two at their limits: 2 / 50. At `opt-level` 0 a document at the limits does not fit
//! in 512 KiB reading, and neither does one at the readers' own limits (24 tables, 12 groups, 65
//! equation levels); no limit both takes what the readers make and fits there.

/// Tables inside tables' cells.
pub(crate) const MAX_TABLES: usize = 32;
/// Groups inside groups.
pub(crate) const MAX_GROUPS: usize = 16;
/// Equation structures inside structures (a fraction in a fraction…), each node counting one.
pub(crate) const MAX_MATH: usize = 80;

/// How deep the walk is, in each kind of nesting.
#[derive(Clone, Copy, Default)]
struct Depth {
    tables: usize,
    groups: usize,
    math: usize,
}

impl Depth {
    /// One table deeper, or `None` past the limit.
    fn table(self) -> Option<Self> {
        (self.tables < MAX_TABLES).then_some(Depth { tables: self.tables + 1, ..self })
    }

    fn group(self) -> Option<Self> {
        (self.groups < MAX_GROUPS).then_some(Depth { groups: self.groups + 1, ..self })
    }

    fn math(self) -> Option<Self> {
        (self.math < MAX_MATH).then_some(Depth { math: self.math + 1, ..self })
    }
}

/// The walk, for one model; the two models are twins.
macro_rules! nesting {
    ($name:ident, $model:ident) => {
        pub(crate) mod $name {
            use std::sync::Arc;

            use super::Depth;
            use m::math::MNode;
            use m::para::InlineObject;
            use $model as m;

            /// Whether every story of `document` nests no deeper than the limits.
            pub(crate) fn within(document: &m::Document) -> bool {
                // The stories that hold blocks. (`the_bridge_names_every_document_field` fails when a
                // field is added to the document.)
                let m::Document { body, parts, footnote_separators, endnote_separators, glossary, .. } = document;
                let start = Depth::default();
                blocks(body, start)
                    && parts.values().all(|part| blocks(&part.blocks, start))
                    && [footnote_separators, endnote_separators].into_iter().all(|separators| {
                        let m::NoteSeparators { separator, continuation_separator, continuation_notice } = separators;
                        [separator, continuation_separator, continuation_notice].into_iter().flatten().all(|story| blocks(story, start))
                    })
                    && glossary.iter().all(|entry| blocks(&entry.blocks, start))
            }

            fn blocks(list: &[Arc<m::Block>], depth: Depth) -> bool {
                list.iter().all(|block| match &**block {
                    m::Block::Para(paragraph) => paragraph.objects.iter().all(|object| self::object(object, depth)),
                    m::Block::Table(table) => {
                        depth.table().is_some_and(|depth| table.rows.iter().all(|row| row.cells.iter().all(|cell| blocks(&cell.blocks, depth))))
                    }
                })
            }

            fn object(object: &InlineObject, depth: Depth) -> bool {
                match object {
                    InlineObject::Group { children, .. } => {
                        depth.group().is_some_and(|depth| children.iter().all(|child| self::object(&child.obj, depth)))
                    }
                    InlineObject::Equation { math, .. } => nodes(&math.nodes, depth),
                    // Named one by one, so that an object which comes to hold others is looked at here.
                    InlineObject::Image { .. }
                    | InlineObject::Graphic { .. }
                    | InlineObject::Shape { .. }
                    | InlineObject::Field { .. }
                    | InlineObject::FieldStart { .. }
                    | InlineObject::FieldEnd
                    | InlineObject::ControlStart { .. }
                    | InlineObject::ControlEnd
                    | InlineObject::NoteRef { .. }
                    | InlineObject::BookmarkStart { .. }
                    | InlineObject::BookmarkEnd { .. }
                    | InlineObject::CommentStart { .. }
                    | InlineObject::CommentEnd { .. }
                    | InlineObject::PermStart { .. }
                    | InlineObject::PermEnd { .. }
                    | InlineObject::Ruby { .. }
                    | InlineObject::PositionalTab { .. }
                    | InlineObject::Opaque { .. } => true,
                }
            }

            fn nodes(arg: &[MNode], depth: Depth) -> bool {
                arg.iter().all(|node| depth.math().is_some_and(|depth| args(node).all(|arg| nodes(arg, depth))))
            }

            /// A node's arguments.
            fn args(node: &MNode) -> impl Iterator<Item = &[MNode]> {
                let (lists, rows): (Vec<&[MNode]>, &[Vec<m::math::Arg>]) = match node {
                    MNode::Run(_) => (Vec::new(), &[]),
                    MNode::Frac { num, den, .. } => (vec![num, den], &[]),
                    MNode::Script { base, sub, sup, .. } => (vec![base, sub, sup], &[]),
                    MNode::Rad { deg, e, .. } => (vec![deg, e], &[]),
                    MNode::Nary { sub, sup, e, .. } => (vec![sub, sup, e], &[]),
                    MNode::Delim { elems, .. } => (elems.iter().map(Vec::as_slice).collect(), &[]),
                    MNode::Func { name, e } => (vec![name, e], &[]),
                    MNode::Lim { e, lim, .. } => (vec![e, lim], &[]),
                    MNode::Acc { e, .. }
                    | MNode::Bar { e, .. }
                    | MNode::BorderBox { e, .. }
                    | MNode::Boxed { e }
                    | MNode::GroupChr { e, .. }
                    | MNode::Phant { e, .. } => (vec![e], &[]),
                    MNode::EqArr { rows } => (rows.iter().map(Vec::as_slice).collect(), &[]),
                    MNode::Matrix { rows, .. } => (Vec::new(), rows.as_slice()),
                };
                lists.into_iter().chain(rows.iter().flatten().map(Vec::as_slice))
            }
        }
    };
}

nesting!(shared, craft_wordmodel);
nesting!(own, wordcraft_doc);
