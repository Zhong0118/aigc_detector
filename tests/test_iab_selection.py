import pytest


def _rows(prefix, count):
    return [{"filename": f"{prefix}_p{i}.png"} for i in range(count)]


def test_select_available_keeps_legacy_and_caps_each_cell():
    from research.image_attribution.iab_selection import select_available

    records = {
        ("a", "cat"): _rows("a-cat", 5),
        ("b", "cat"): _rows("b-cat", 2),
        ("real", "cat"): _rows("real-cat", 4),
        ("a", "dog"): _rows("a-dog", 5),
        ("b", "dog"): _rows("b-dog", 5),
        ("real", "dog"): _rows("real-dog", 3),
    }
    legacy = {
        "a/cat": records[("a", "cat")][:1],
        "b/cat": records[("b", "cat")][:1],
        "real/cat": records[("real", "cat")][:1],
        "a/dog": records[("a", "dog")][:1],
        "b/dog": records[("b", "dog")][:1],
        "real/dog": records[("real", "dog")][:1],
    }

    selected, audit = select_available(records, legacy, per_cell=3, seed=17)

    assert selected[("a", "cat")][0] == legacy["a/cat"][0]
    assert selected[("b", "cat")] == records[("b", "cat")]
    assert selected[("a", "cat")] == select_available(records, legacy, 3, 17)[0][("a", "cat")]
    assert all(len(selected[(source, semantic)]) <= 3
               for source, semantic in selected if source != "real")
    assert len(selected[("real", "cat")]) == 4
    assert len(selected[("real", "dog")]) == 3
    assert audit["ai_selected"] == 11
    assert audit["real_selected"] == 7
    assert audit["short_cells"] == {"b/cat": 2}


def test_select_available_rejects_legacy_items_missing_from_verified_catalog():
    from research.image_attribution.iab_selection import select_available

    records = {("model", "cat"): _rows("new", 3), ("real", "cat"): _rows("real", 3)}
    legacy = {"model/cat": [{"filename": "old_p1.png"}]}

    with pytest.raises(ValueError, match="legacy selection is unavailable"):
        select_available(records, legacy, per_cell=2, seed=1)



def test_selection_audit_totals_selected_member_bytes():
    from research.image_attribution.iab_selection import select_available

    records = {
        ("a", "cat"): [{"filename": "a.png", "size": 11, "compressed_size": 7}],
        ("b", "cat"): [{"filename": "b.png", "size": 13, "compressed_size": 9}],
        ("real", "cat"): [{"filename": "r.png", "size": 17, "compressed_size": 12}],
    }

    _, audit = select_available(records, per_cell=1, seed=1)

    assert audit["selected_uncompressed_bytes"] == 41
    assert audit["selected_compressed_bytes"] == 28
