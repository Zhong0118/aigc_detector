import pytest


def _sample(sample_id, source, semantic):
    return {
        "sample_id": sample_id,
        "source_label": source if source != "real" else None,
        "semantic_class": semantic,
    }


def test_receipt_validation_accepts_availability_capped_cells():
    from scripts.prepare_iab_manifest import validate_receipts

    selection = {
        "entries": {"model/cat": [{}, {}], "real/cat": [{}, {}, {}]},
        "audit": {"selected_by_cell": {"model/cat": 2, "real/cat": 3}},
    }
    status = {"selected": 5, "selected_by_cell": {"model/cat": 2, "real/cat": 3}}
    receipts = [
        {"source": "model", "semantic": "cat", "selected": 2,
         "rows": [_sample("a", "model", "cat"), _sample("b", "model", "cat")]},
        {"source": "real", "semantic": "cat", "selected": 3,
         "rows": [_sample("r1", "real", "cat"), _sample("r2", "real", "cat"),
                  _sample("r3", "real", "cat")]},
    ]

    rows = validate_receipts(selection, status, receipts)

    assert len(rows) == 5


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "wrong_cell"])
def test_receipt_validation_rejects_inconsistent_cells(mutation):
    from scripts.prepare_iab_manifest import validate_receipts

    selection = {
        "entries": {"model/cat": [{}, {}], "real/cat": [{}, {}]},
        "audit": {"selected_by_cell": {"model/cat": 2, "real/cat": 2}},
    }
    status = {"selected": 4, "selected_by_cell": {"model/cat": 2, "real/cat": 2}}
    rows = [_sample("a", "model", "cat"), _sample("b", "model", "cat")]
    receipts = [
        {"source": "model", "semantic": "cat", "selected": 2, "rows": rows},
        {"source": "real", "semantic": "cat", "selected": 2,
         "rows": [_sample("r1", "real", "cat"), _sample("r2", "real", "cat")]},
    ]
    if mutation == "missing":
        receipts.pop()
        status["selected"] = 2
    elif mutation == "duplicate":
        receipts[1]["rows"][0]["sample_id"] = "a"
    else:
        receipts[1]["rows"][0]["semantic_class"] = "dog"

    with pytest.raises(ValueError):
        validate_receipts(selection, status, receipts)
