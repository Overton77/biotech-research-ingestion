# Diagnosis: Windows path error and ResearchMission checkpoint serialization

## Summary

Two separate issues occurred during a long deep-research run (second task, ~20 min in):

1. **Path ValueError** in the subagent filesystem backend (Windows).
2. **TypeError: ResearchMission not msgpack serializable** when LangGraph tried to checkpoint state to Postgres.

Both are fixable in this repo without waiting for upstream changes.

---

## Issue A: Path “outside root directory” on Windows

### What happened

```
ValueError: Path:\\?\C:\...\secondary-source-finder\outputs\task-2\raw\qualia-podcast-decoded.txt
outside root directory: C:\...\secondary-source-finder
```

The subagent `secondary-source-finder` uses a `FilesystemBackend` with:

- **Root:** `.../tasks/task-2/subagents/secondary-source-finder`
- **Requested path:** `outputs/task-2/raw/qualia-podcast-decoded.txt` (logically under that root)

So the path is logically inside the root, but the backend still raises “outside root”.

### Root cause

In `deepagents.backends.filesystem.FilesystemBackend._resolve_path` (with `virtual_mode=True`):

1. `full = (self.cwd / vpath.lstrip("/")).resolve()`
2. Containment is checked with `full.relative_to(self.cwd)`.

On Windows, `Path.resolve()` can return a path with the long-path prefix `\\?\`. The stored `self.cwd` comes from `Path(root_dir).resolve()` and may not use that prefix. So:

- `full` = `\\?\C:\...\secondary-source-finder\outputs\task-2\raw\...`
- `self.cwd` = `C:\...\secondary-source-finder`

`pathlib` treats these as different path forms, so `full.relative_to(self.cwd)` raises `ValueError` even though the path is logically under the root. This is a known class of Windows path-normalization bugs (see [Backends — FilesystemBackend](https://docs.langchain.com/oss/python/deepagents/backends)).

### Fix (in this repo)

We cannot change the deepagents package here. So we add a **Windows-safe wrapper** in `backends.py` that normalizes both paths before the containment check:

- Subclass `FilesystemBackend`.
- Override `_resolve_path` to:
  - Compute `full` as today.
  - Normalize `full` and `self.cwd` to the same form (e.g. strip `\\?\` and use `os.path.normpath` / same `Path` normalization) so that containment is checked on comparable paths.
  - Then call `relative_to` on the normalized pair (or equivalent logic) and return the resolved path.

Use this backend (or a factory that returns it) for both task and subagent filesystem backends so all runs on Windows use the same, safe resolution.

---

## Issue B: ResearchMission not msgpack serializable

### What happened

When the runner tried to checkpoint state to Postgres:

```
TypeError: Type is not msgpack serializable: ResearchMission
```

So the graph state contained something the LangGraph checkpointer (msgpack) cannot serialize.

### Root cause

- The mission runner is compiled with a **Postgres checkpointer** (`get_deep_agents_persistence()`).
- The runner state type `MissionRunnerState` includes:
  - `mission: Any`  # actually a Beanie `ResearchMission` document
- `load_mission` returns `{"mission": mission}`, so the full Beanie document is written into graph state.
- After each step, LangGraph serializes the full state for checkpointing; Beanie/Pydantic document types are not msgpack-serializable, so serialization fails.

So the problem is not a “malformed ResearchRun” but **non-serializable state**: the live `ResearchMission` (and possibly other complex types) in state.

### Fix (in this repo)

Make checkpointed state **only contain serializable data** (e.g. built-in types, dict/list that are JSON-/msgpack-friendly):

1. **Remove `mission` from state**
   - Do not return `mission` from `load_mission` (or any node).
   - In every node that currently uses `state["mission"]`, load the mission by id:  
     `mission = await ResearchMission.get(PydanticObjectId(state["mission_id"]))`.
   - Nodes that need this: `initialize_runtime_state`, `compute_ready_queue`, `run_task`, `persist_research_run`, `finalize_mission`.

2. **Keep mission-derived data in state in serializable form**
   - From `load_mission`, return only serializable fields:
     - `task_defs_by_id`: store as `dict[str, dict]` (e.g. `{tid: td.model_dump() for td in mission.task_defs}`) so we never put Pydantic/Beanie objects in state.
     - `dependency_map`, `reverse_dependency_map` (already dicts).
     - Optional: `task_def_order = [td.task_id for td in mission.task_defs]` for stable ordering in `compute_ready_queue` without using `mission.task_defs`.
   - Where code expects a `TaskDef` object, do `TaskDef.model_validate(state["task_defs_by_id"][task_id])` (or equivalent) inside the node.

3. **Ensure no other non-serializable types slip into state**
   - If later you see serialization errors for other types (e.g. `TaskResult`, `ArtifactRef`), store them as dicts (e.g. `model_dump()` / `model_validate()`) or exclude them from checkpointed state the same way.

After this, the runner can run long missions with Postgres checkpointer without hitting “ResearchMission not msgpack serializable”, and the path fix above will prevent the Windows “outside root” error in subagent filesystem writes.

---

## References

- [Backends — FilesystemBackend (virtual_mode, root_dir)](https://docs.langchain.com/oss/python/deepagents/backends)
- `deepagents.backends.filesystem.FilesystemBackend._resolve_path` (virtual_mode branch)
- Mission runner: `MissionRunnerState`, `build_mission_runner()`, `get_deep_agents_persistence()`
