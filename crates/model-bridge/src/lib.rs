//! WordCraft's document model to and from `craft-wordmodel`, the model the shared format crates
//! (`craft-docx`, `craft-doc-binary`) read into and write from.
//!
//! The two models are twins: `craft-wordmodel` was taken out of `wordcraft-doc`, and almost every
//! type has the same members in both. They are still different Rust types, so a document read by
//! a shared crate has to be converted. This crate does it member by member with `craft-convert`:
//! `mapping.rs` lists the members of each shared type, `stories.rs` holds the document, its
//! stories and the few types written by hand. The lists are checked by the compiler, so a member
//! added to either model stops the build here instead of being dropped.
//!
//! It replaced a bridge that sent every leaf value through `serde_json`. What that bridge did
//! with a value is kept, including where it was strict: a number that is not finite is an error
//! ([`Error::NotFinite`]) unless it is optional, in which case it becomes `None`. A document nested
//! deeper than the conversion can recurse is an error too ([`Error::TooDeep`]; see `nesting.rs`).
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::unimplemented, clippy::todo, clippy::unreachable)]
// `convert_types!` walks members one at a time; the widest structs here have more than sixty.
#![recursion_limit = "256"]

mod json;
mod mapping;
mod nesting;
mod session;
mod stories;

use craft_convert::{Convert, ConvertRef, Floats};

/// Names this crate's mapping: neither model is this crate's own, so its conversions are
/// `Convert<_, Bridge>`.
pub(crate) struct Bridge;

/// Every number converted is looked at: the shared format crates and the layout cannot take NaN
/// or an infinity, and the JSON bridge refused them (JSON cannot write one).
impl Floats for Bridge {
    fn f32(value: f32) -> f32 {
        if !value.is_finite() {
            session::not_finite();
        }
        value
    }

    fn f64(value: f64) -> f64 {
        if !value.is_finite() {
            session::not_finite();
        }
        value
    }
}

/// Why a conversion was refused.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Error {
    /// A number that is not finite was in a place where it cannot be dropped or replaced.
    NotFinite,
    /// Tables, groups or equation structures nest deeper than the conversion takes.
    TooDeep,
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Error::NotFinite => "shared model conversion: a number is not finite",
            Error::TooDeep => "shared model conversion: tables, groups or equations are nested too deep",
        })
    }
}

impl std::error::Error for Error {}

/// A document read by a shared format crate, as WordCraft's. The source is consumed: text and
/// formatting are moved, and pictures and opaque parts keep their buffers.
///
/// `fit_state` is left empty; it is not part of the shared model (`wordcraft-docx` reads it from
/// a custom XML part).
///
/// A document nested deeper than the limits in `nesting.rs` is refused, and leaked: see below.
pub fn from_shared(source: craft_wordmodel::Document) -> Result<wordcraft_doc::Document, Error> {
    if !nesting::shared::within(&source) {
        // Dropping it would recurse as deep as it nests and overflow the stack, which is what
        // refusing it prevents. No reader makes such a document (they stop well inside the
        // limits), so leaking it is the safe outcome for a value that should not exist, and costs
        // less than code to take it apart. `to_shared` borrows, so a refused document there stays
        // the caller's.
        std::mem::forget(source);
        return Err(Error::TooDeep);
    }
    session::checked(|| stories::document_from_shared(source))
}

/// WordCraft's document for a shared format crate to write. Pictures and opaque parts share
/// their buffers with `source`. A document nested deeper than the limits in `nesting.rs` is
/// refused.
pub fn to_shared(source: &wordcraft_doc::Document) -> Result<craft_wordmodel::Document, Error> {
    if !nesting::own::within(source) {
        return Err(Error::TooDeep);
    }
    session::checked(|| stories::document_to_shared(source))
}

/// A chart's description, for the shared crate to draw or write.
pub fn chart_to_shared(spec: &wordcraft_doc::chart::ChartSpec) -> Result<craft_wordmodel::chart::ChartSpec, Error> {
    session::checked(|| ConvertRef::<_, Bridge>::convert_ref(spec))
}

/// A SmartArt description, for the shared crate to draw.
pub fn smart_art_to_shared(spec: &wordcraft_doc::smart_art::SmartArtSpec) -> Result<craft_wordmodel::smart_art::SmartArtSpec, Error> {
    session::checked(|| ConvertRef::<_, Bridge>::convert_ref(spec))
}

/// Theme colours, for the shared crate to draw with.
pub fn theme_to_shared(theme: &[wordcraft_doc::Rgb]) -> Vec<craft_wordmodel::props::Rgb> {
    ConvertRef::<_, Bridge>::convert_ref(theme)
}

/// What the shared crate drew for a chart or SmartArt.
pub fn graphic_items_from_shared(items: Vec<craft_wordmodel::graphic::GraphicItem>) -> Result<Vec<wordcraft_doc::graphic::GraphicItem>, Error> {
    session::checked(|| Convert::<_, Bridge>::convert(items))
}

/// Bibliography sources read by the shared crate.
pub fn sources_from_shared(sources: Vec<craft_wordmodel::Source>) -> Result<Vec<wordcraft_doc::Source>, Error> {
    session::checked(|| Convert::<_, Bridge>::convert(sources))
}

#[cfg(test)]
mod tests;
