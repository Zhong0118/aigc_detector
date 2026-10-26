"""CPU metadata regressions for the unused future extension split protocol."""
from collections import Counter
import copy
import hashlib
import random
import unittest

from research.image_attribution.iab_protocol import group_and_split
from research.image_attribution.extension_split import group_and_split_extension_v2


def row(sid, **overrides):
    result = dict(sample_id=sid, sha256='bytes-' + sid, pixel_sha256='pixels-' + sid,
                  phash=hashlib.sha256(sid.encode()).hexdigest()[:16], is_ai=1,
                  source_label='source-' + sid, semantic_class='semantic-' + sid,
                  prompt_id='prompt-' + sid)
    result.update(overrides)
    return result


class ExtensionSplitTests(unittest.TestCase):
    def test_singleton_strata_are_not_all_holdout(self):
        inputs = [row(f'new-{i:04d}') for i in range(300)]
        legacy, _ = group_and_split(inputs)
        self.assertEqual(Counter(r['split'] for r in legacy), {'holdout': 300})
        result, audit = group_and_split_extension_v2(inputs)
        counts = Counter(r['split'] for r in result)
        self.assertEqual(len({r['group_id'] for r in result}), 300)
        self.assertTrue(180 < counts['train'] < 240)
        self.assertTrue(20 < counts['dev'] < 70)
        self.assertTrue(20 < counts['holdout'] < 70)
        self.assertEqual(audit['protocol'], 'future-extension-split-v2')
        self.assertEqual(audit['allocation_unit'], 'unanchored_group')
        self.assertEqual(audit['expected_group_ratios'], {'train': .7, 'dev': .15, 'holdout': .15})
        self.assertFalse(audit['hard_counts'])
        self.assertEqual(audit['unanchored_group_split_counts'], dict(counts))

    def test_deterministic_for_input_order_and_unrelated_additions(self):
        inputs = [row(f'new-{i:04d}') for i in range(100)]
        expected = group_and_split_extension_v2(inputs, seed=42)
        shuffled = copy.deepcopy(inputs)
        random.Random(99).shuffle(shuffled)
        self.assertEqual(expected, group_and_split_extension_v2(shuffled, seed=42))
        self.assertEqual(expected, group_and_split_extension_v2(inputs[::-1], seed=42))
        expanded, _ = group_and_split_extension_v2(inputs + [row('unrelated')], seed=42)
        old_ids = {r['sample_id'] for r in inputs}
        self.assertEqual(expected[0], [r for r in expanded if r['sample_id'] in old_ids])

    def test_parent_rows_and_anchored_new_members_are_unchanged(self):
        parents = [row(f'old-{split}', split=split, group_id=f'frozen-{split}')
                   for split in ('train', 'dev', 'holdout')]
        parents.append(row('old-real', split='holdout', group_id='frozen-real',
                           is_ai=0, source_label=None, prompt_id=None))
        new = [row('new-' + p['sample_id'], semantic_class=p['semantic_class'],
                   prompt_id=p['prompt_id'], is_ai=p['is_ai'], source_label=p['source_label'],
                   pixel_sha256=p['pixel_sha256'] if not p['is_ai'] else 'new-pixel-' + p['sample_id'])
               for p in parents]
        inputs = parents + new + [row('free-new')]
        snapshot = copy.deepcopy(inputs)
        result, audit = group_and_split_extension_v2(inputs, frozen_rows=parents)
        by_id = {r['sample_id']: r for r in result}
        for parent in parents:
            self.assertEqual(by_id[parent['sample_id']], parent)
            child = by_id['new-' + parent['sample_id']]
            self.assertEqual((child['group_id'], child['split']), (parent['group_id'], parent['split']))
        self.assertEqual(inputs, snapshot)
        self.assertEqual(audit['anchored_groups'], 4)
        self.assertEqual(audit['unanchored_groups'], 1)

    def test_unanchored_near_duplicates_stay_in_one_group(self):
        inputs = [row('a', phash='0000000000000000'), row('b', phash='000000000000000f')]
        result, audit = group_and_split_extension_v2(inputs)
        self.assertEqual(len({(r['group_id'], r['split']) for r in result}), 1)
        self.assertEqual(audit['unanchored_groups'], 1)
        self.assertEqual(audit['near_duplicate_links'][0]['phash_distance'], 4)

    def test_new_bridge_still_rejects_frozen_split_and_group_conflicts(self):
        for split, message in [('holdout', 'frozen split conflict'), ('train', 'frozen group conflict')]:
            with self.subTest(split=split):
                parents = [row('a', phash='0000000000000000', split='train', group_id='parent-a'),
                           row('b', phash='00000000000000ff', split=split, group_id='parent-b')]
                bridge = row('bridge', phash='000000000000000f')
                with self.assertRaisesRegex(ValueError, message):
                    group_and_split_extension_v2(parents + [bridge], frozen_rows=parents)

    def test_changed_parent_identity_still_rejected(self):
        parents = [row('a', split='holdout', group_id='parent-a')]
        changed = [dict(parents[0], source_label='changed')]
        with self.assertRaisesRegex(ValueError, 'frozen sample changed'):
            group_and_split_extension_v2(changed, frozen_rows=parents)


if __name__ == '__main__':
    unittest.main()
