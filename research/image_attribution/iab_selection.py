"""Deterministic availability-capped IAB selection that extends a legacy manifest."""
import random


def select_available(records, legacy_entries=None, per_cell=200, seed=42):
    """Keep legacy members, then fill each AI cell up to its available cap.

    The real-image cap is per_cell times the number of AI sources per semantic,
    bounded by the catalog. No sample is duplicated to satisfy a target.
    """
    if per_cell <= 0:
        raise ValueError("per_cell must be positive")
    legacy_entries = legacy_entries or {}
    sources = sorted({source for source, _ in records if source != "real"})
    semantics = sorted({semantic for _, semantic in records})
    if not sources or len(records) != (len(sources) + 1) * len(semantics):
        raise ValueError("catalog must contain every source/semantic cell and real data")

    selected = {}
    short_cells = {}
    real_short_semantics = {}
    for semantic in semantics:
        for source in sources:
            key = (source, semantic)
            available = records[key]
            by_name = {row["filename"]: row for row in available}
            legacy = list(legacy_entries.get(f"{source}/{semantic}", []))
            legacy_names = [row["filename"] for row in legacy]
            if len(set(legacy_names)) != len(legacy_names):
                raise ValueError(f"duplicate legacy entries for {source}/{semantic}")
            if any(name not in by_name for name in legacy_names):
                raise ValueError(f"legacy selection is unavailable for {source}/{semantic}")
            cap = min(per_cell, len(available))
            if len(legacy) > cap:
                raise ValueError(f"legacy selection exceeds available cap for {source}/{semantic}")
            legacy_names = set(legacy_names)
            remainder = [row for row in available if row["filename"] not in legacy_names]
            rng = random.Random(f"{seed}:{source}:{semantic}")
            selected[key] = legacy + rng.sample(remainder, cap - len(legacy))
            if cap < per_cell:
                short_cells[f"{source}/{semantic}"] = cap

        key = ("real", semantic)
        available = records[key]
        by_name = {row["filename"]: row for row in available}
        legacy = list(legacy_entries.get(f"real/{semantic}", []))
        legacy_names = [row["filename"] for row in legacy]
        if len(set(legacy_names)) != len(legacy_names):
            raise ValueError(f"duplicate legacy entries for real/{semantic}")
        if any(name not in by_name for name in legacy_names):
            raise ValueError(f"legacy selection is unavailable for real/{semantic}")
        target = min(per_cell * len(sources), len(available))
        if len(legacy) > target:
            raise ValueError(f"legacy selection exceeds available cap for real/{semantic}")
        legacy_names = set(legacy_names)
        remainder = [row for row in available if row["filename"] not in legacy_names]
        rng = random.Random(f"{seed}:real:{semantic}")
        selected[key] = legacy + rng.sample(remainder, target - len(legacy))
        if target < per_cell * len(sources):
            real_short_semantics[semantic] = dict(selected=target, available=len(available),
                                                  requested=per_cell * len(sources))

    selected_rows = [row for rows in selected.values() for row in rows]
    audit = dict(
        requested_ai_per_source_semantic=per_cell,
        ai_selected=sum(len(rows) for (source, _), rows in selected.items() if source != "real"),
        real_requested_per_semantic=per_cell * len(sources),
        real_selected=sum(len(selected[("real", semantic)]) for semantic in semantics),
        short_cells=short_cells,
        real_short_semantics=real_short_semantics,
        selected_by_cell={f"{source}/{semantic}": len(rows)
                          for (source, semantic), rows in sorted(selected.items())},
        selected_uncompressed_bytes=sum(int(row.get("size", 0)) for row in selected_rows),
        selected_compressed_bytes=sum(int(row.get("compressed_size", row.get("size", 0)))
                                      for row in selected_rows),
    )
    return selected, audit



def select_prefix(records, parent_entries, per_cell, seed=42):
    """Reduce a frozen parent selection without introducing or reshuffling members."""
    if per_cell <= 0:
        raise ValueError('per_cell must be positive')
    sources = {source for source, _ in records if source != 'real'}
    if set(parent_entries) != {f'{source}/{semantic}' for source, semantic in records}:
        raise ValueError('parent selection cell coverage differs')
    prefix = {}
    for (source, semantic), available in records.items():
        key = f'{source}/{semantic}'
        count = min(len(available), per_cell * (len(sources) if source == 'real' else 1))
        parent = parent_entries[key]
        if len(parent) < count:
            raise ValueError(f'parent selection lacks required members: {key}')
        prefix[key] = parent[:count]
    return select_available(records, prefix, per_cell, seed)
