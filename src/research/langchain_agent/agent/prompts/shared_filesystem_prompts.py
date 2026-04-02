FILESYSTEM_TOOL_DESCRIPTIONS = {
    "ls": (
        "List directories and files inside the shared sandbox. "
        "Use relative paths like runs, reports, and scratch."
    ),
    "read_file": (
        "Read saved notes, intermediate findings, and handoff artifacts from "
        "sandbox-relative paths."
    ),
    "write_file": (
        "Create structured intermediate files and final reports inside the shared "
        "sandbox only. Prefer markdown or json."
    ),
    "edit_file": (
        "Update an existing findings file, handoff artifact, or report incrementally "
        "inside the shared sandbox."
    ),
}

FILESYSTEM_SYSTEM_PROMPT = (
    "Use the filesystem intelligently for intermediate research state. "
    "All paths must stay inside the sandbox root. "
    "Use only relative sandbox paths such as runs/, reports/, and scratch/. "
    "Never use absolute host paths."
)