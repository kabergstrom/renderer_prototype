use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

#[derive(Copy, Clone, PartialEq, Debug)]
pub enum IncludeType {
    Relative,
    Standard,
}

impl From<shaderc::IncludeType> for IncludeType {
    fn from(include_type: shaderc::IncludeType) -> Self {
        match include_type {
            shaderc::IncludeType::Relative => IncludeType::Relative,
            shaderc::IncludeType::Standard => IncludeType::Standard,
        }
    }
}

pub struct ResolvedInclude {
    pub resolved_path: PathBuf,
    pub content: String,
}

impl Into<shaderc::ResolvedInclude> for ResolvedInclude {
    fn into(self) -> shaderc::ResolvedInclude {
        shaderc::ResolvedInclude {
            content: self.content,
            resolved_name: self.resolved_path.to_str().unwrap().to_string(),
        }
    }
}

/// Where `#include` files come from. Both the compiler (shaderc) and the
/// reflection parser resolve through the same value, so they see the same
/// files.
#[derive(Clone, Debug)]
pub enum Includes {
    /// The filesystem: relative to the including file, then each directory
    /// (`--include-path`) in order.
    Ambient(Arc<[PathBuf]>),
    /// In-memory files by virtual path (`/`-separated): what a
    /// dependency-tracking caller (an asset importer) read. Resolved by
    /// [`virtual_include_candidates`].
    Memory(Arc<BTreeMap<String, String>>),
}

impl Default for Includes {
    fn default() -> Self {
        Includes::Ambient(Arc::from(Vec::new()))
    }
}

/// The virtual paths an `#include "requested"` in the file at virtual path
/// `from` may name, in the order they are tried: relative to `from`'s
/// directory, then `requested` itself (a path from the namespace's root).
/// A candidate that climbs above the root is skipped. Callers that collect
/// includes for [`Includes::Memory`] must resolve with this same rule.
pub fn virtual_include_candidates(from: &str, requested: &str) -> Vec<String> {
    let dir = from.rfind('/').map_or("", |slash| &from[..=slash]);
    let mut candidates = Vec::with_capacity(2);
    for joined in [format!("{dir}{requested}"), requested.to_owned()] {
        if let Some(path) = normalize_virtual(&joined) {
            if !candidates.contains(&path) {
                candidates.push(path);
            }
        }
    }
    candidates
}

/// `path` with `.` and `..` segments folded; `None` when it climbs above the
/// root.
fn normalize_virtual(path: &str) -> Option<String> {
    let mut segments: Vec<&str> = Vec::new();
    for segment in path.split('/') {
        match segment {
            "" | "." => {}
            ".." => {
                segments.pop()?;
            }
            segment => segments.push(segment),
        }
    }
    Some(segments.join("/"))
}

impl Includes {
    pub(crate) fn resolve(
        &self,
        requested_path: &Path,
        include_type: IncludeType,
        requested_from: &Path,
        include_depth: usize,
    ) -> Result<ResolvedInclude, String> {
        log::trace!(
            "include file {:?} {:?} {:?} {:?}",
            requested_path,
            include_type,
            requested_from,
            include_depth
        );
        match self {
            Includes::Ambient(search) => ambient(requested_path, requested_from, search),
            Includes::Memory(files) => {
                let (from, requested) = (
                    requested_from.to_string_lossy(),
                    requested_path.to_string_lossy(),
                );
                virtual_include_candidates(&from, &requested)
                    .into_iter()
                    .find_map(|path| {
                        files.get(&path).map(|content| ResolvedInclude {
                            resolved_path: PathBuf::from(path),
                            content: content.clone(),
                        })
                    })
                    .ok_or_else(|| {
                        format!(
                            "include {requested:?} from {from:?} is not among the files the caller read"
                        )
                    })
            }
        }
    }

    /// The include callback shaderc compiles with. Each file is included
    /// once per compile, as the reflection parser reads it (the shared
    /// headers carry no include guards): its content comes wrapped in a guard
    /// macro named for its resolved path. A guard, not a seen-set, since
    /// shaderc runs the includes twice (preprocess, then compile).
    pub(crate) fn shaderc_callback(
        &self
    ) -> impl Fn(&str, shaderc::IncludeType, &str, usize) -> shaderc::IncludeCallbackResult + 'static
    {
        let includes = self.clone();
        move |requested, include_type, from, depth| {
            let mut resolved = includes.resolve(
                Path::new(requested),
                include_type.into(),
                Path::new(from),
                depth,
            )?;
            let guard: String = resolved
                .resolved_path
                .to_string_lossy()
                .chars()
                .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
                .collect();
            resolved.content = format!(
                "#ifndef RAFX_INCLUDED_{guard}\n#define RAFX_INCLUDED_{guard}\n#line 1\n{}\n#endif\n",
                resolved.content
            );
            Ok(resolved.into())
        }
    }

    pub(crate) fn is_ambient(&self) -> bool {
        matches!(self, Includes::Ambient(_))
    }
}

/// Filesystem include: relative to the including file (or absolute), then
/// under each `search` directory.
fn ambient(
    requested_path: &Path,
    requested_from: &Path,
    search: &[PathBuf],
) -> Result<ResolvedInclude, String> {
    let local_path = if requested_path.is_absolute() {
        requested_path.to_path_buf()
    } else {
        requested_from
            .parent()
            .unwrap_or(Path::new(""))
            .join(requested_path)
    };
    let resolved_path = if local_path.is_file() || requested_path.is_absolute() {
        local_path
    } else {
        search
            .iter()
            .map(|dir| dir.join(requested_path))
            .find(|path| path.is_file())
            .unwrap_or(local_path)
    };
    let content = std::fs::read_to_string(&resolved_path).map_err(|e| {
        format!(
            "Could not read file {:?} when trying to include {:?} from {:?}: {:?}",
            resolved_path, requested_path, requested_from, e
        )
    })?;
    Ok(ResolvedInclude {
        resolved_path,
        content,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn virtual_includes_try_the_including_directory_then_the_root() {
        assert_eq!(
            virtual_include_candidates("fx/a.frag", "common.glsl"),
            ["fx/common.glsl", "common.glsl"]
        );
        assert_eq!(
            virtual_include_candidates("fx/a.frag", "../lib/x.glsl"),
            ["lib/x.glsl"]
        );
        assert_eq!(virtual_include_candidates("a.frag", "x.glsl"), ["x.glsl"]);
        assert!(virtual_include_candidates("a.frag", "../x.glsl").is_empty());
    }
}
