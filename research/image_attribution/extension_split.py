"""Future expansion protocol v2 only; not wired into the current v6 pipeline."""
from collections import Counter
import hashlib

from research.image_attribution.iab_protocol import group_and_split


def group_and_split_extension_v2(input_rows, seed=42, frozen_rows=None):
    """Preserve parent anchors; hash only new unanchored groups into 70/15/15.

    Ratios are expectations over groups, not exact group or sample counts.
    Existing grouping and frozen identity/split/group conflict checks still apply.
    """
    rows, audit = group_and_split(input_rows, seed=seed, frozen_rows=frozen_rows)
    anchored = {r['group_id'] for r in frozen_rows} if frozen_rows is not None else set()
    groups = {r['group_id'] for r in rows}
    assignments = {}
    for group in sorted(groups - anchored):
        value = int.from_bytes(hashlib.sha256(f'iab-extension-v2:{seed}:{group}'.encode()).digest(), 'big')
        assignments[group] = ('train' if value * 100 < 70 * 2**256 else
                              'dev' if value * 100 < 85 * 2**256 else 'holdout')
    for row in rows:
        if row['group_id'] in assignments:
            row['split'] = assignments[row['group_id']]
    group_splits = {r['group_id']: r['split'] for r in rows}
    audit.update(
        protocol='future-extension-split-v2', scope='future_expansion_only', seed=seed,
        split_policy='preserve parent splits/groups and anchored new members; domain-separated SHA256 for unanchored groups',
        allocation_unit='unanchored_group', expected_group_ratios={'train': .7, 'dev': .15, 'holdout': .15},
        hard_counts=False, ratio_note='Group-level expected ratios; no exact counts or sample-level ratio guarantee.',
        hash_rule='UTF-8 iab-extension-v2:seed:group_id; SHA256 unsigned big-endian integer / 2**256; thresholds 0.70, 0.85',
        anchored_groups=len(anchored), unanchored_groups=len(assignments),
        unanchored_group_split_counts=dict(Counter(assignments.values())),
        group_split_counts=dict(Counter(group_splits.values())),
        sample_split_counts=dict(Counter(r['split'] for r in rows)))
    return rows, audit
