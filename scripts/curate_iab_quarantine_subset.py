#!/usr/bin/env python3
"""Derive an exclusive offline IAB subset using a hash-bound, approved exclusion list."""
import argparse
import copy
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research.image_attribution.extension_split import group_and_split_extension_v2
from research.image_attribution.protocol import sha256
from scripts.prepare_iab_extension_manifest import selected_receipts


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2); stream.write("\n")


def curate(root, config_path):
    root = Path(root).resolve(); config_path = Path(config_path)
    cfg = json.loads(config_path.read_text())
    def local(name):
        value = Path(name)
        if value.is_absolute() or ".." in value.parts or not (root / value).resolve().is_relative_to(root):
            raise ValueError("curation path must remain within workspace")
        return root / value
    def bound(spec, filename=None):
        path = local(spec['path'])
        if filename: path /= filename
        expected = spec[filename.replace('.json','') + '_sha256'] if filename else spec['sha256']
        if sha256(path) != expected: raise ValueError("bound input hash changed: " + str(path))
        return path
    source = local(cfg['source']['path']); output = local(cfg['output'])
    if output.exists(): raise FileExistsError("curated output exists; preserve it")
    if output.resolve().is_relative_to(source.resolve()) or source.resolve().is_relative_to(output.resolve()):
        raise ValueError("source and output must be independent workspace directories")
    selection_path = bound(cfg['source'], 'selection.json')
    status_path = local(cfg['source']['path']) / 'download_status.json'
    if sha256(status_path) != cfg['source']['status_sha256']: raise ValueError('source status hash changed')
    parent_path = bound(cfg['parent_manifest']); excluded_path = bound(cfg['quarantine'])
    excluded_list = json.loads(excluded_path.read_text()); excluded = set(excluded_list)
    if len(excluded) != len(excluded_list) or len(excluded) != cfg['quarantine']['count']:
        raise ValueError('duplicate or unexpected exclusion count')
    selection = json.loads(selection_path.read_text()); status = json.loads(status_path.read_text())
    if status.get('status') != 'completed': raise ValueError('source acquisition incomplete')
    receipt_paths = sorted((source/'receipts').glob('*.json'))
    receipts = [json.loads(path.read_text()) for path in receipt_paths]
    rows = selected_receipts(selection, status, receipts)
    frozen = [json.loads(line) for line in parent_path.read_text().splitlines() if line.strip()]
    parent_ids = {row['sample_id'] for row in frozen}; all_ids = {row['sample_id'] for row in rows}
    if excluded & parent_ids: raise ValueError('exclusion intersects frozen parent')
    if not excluded <= all_ids: raise ValueError('exclusion contains unknown sample')
    kept = [row for row in rows if row['sample_id'] not in excluded]
    if (len(rows) != cfg['expected_source_samples'] or len(kept) != cfg['expected_output_samples'] or
        len(receipts) != cfg['expected_cells']): raise ValueError('registered sample/cell count mismatch')
    _, grouping = group_and_split_extension_v2(kept, cfg['seed'], frozen_rows=frozen)
    # Full byte/pixel/pHash integrity and immutable manifest creation remain owned by the anchored preparer.
    paths = []
    for row in kept:
        relative = Path(row['relative_path'])
        if relative.is_absolute() or '..' in relative.parts: raise ValueError('image path escapes workspace subset')
        for rel in (relative, relative.with_suffix('.json')):
            path = source / rel
            if not path.resolve().is_relative_to(source.resolve()): raise ValueError('image escapes source subset')
            if not path.is_file(): raise FileNotFoundError(path)
            paths.append(rel)
    entries = copy.deepcopy(selection['entries'])
    allowed = {(row['source_label'] or 'real', row['semantic_class'], row['source_evidence']['zip_member']) for row in kept}
    for cell in entries:
        source_label, semantic = cell.split('/', 1)
        entries[cell] = [entry for entry in entries[cell] if (source_label, semantic, entry['filename']) in allowed]
    counts = {cell:len(values) for cell,values in entries.items()}
    ai = sum(row['is_ai'] for row in kept); real = len(kept)-ai
    filtered = copy.deepcopy(selection); filtered['entries'] = entries
    filtered.update(selection_policy='fixed published prefix minus explicitly approved new conflict members', reused_from=cfg['source']['path'])
    filtered['audit'].update(selected_by_cell=counts, ai_selected=ai, real_selected=real,
        selected_uncompressed_bytes=sum(e.get('size',0) for values in entries.values() for e in values),
        selected_compressed_bytes=sum(e.get('compressed_size',0) for values in entries.values() for e in values),
        curation_excluded_by_cell={cell:len(selection['entries'][cell])-counts[cell] for cell in counts if counts[cell]!=len(selection['entries'][cell])})
    derived_receipts=[]
    for path,receipt in zip(receipt_paths,receipts):
        cell_rows=[row for row in receipt['rows'] if row['sample_id'] not in excluded]
        derived_receipts.append((path.name,dict(source=receipt['source'],semantic=receipt['semantic'],
            selected=len(cell_rows),rows=cell_rows,range_bytes=0,network=dict(bytes=0,requests_attempted=0),
            derivation='offline hardlink subset; original acquisition evidence retained in source receipts',source_receipt_sha256=sha256(path))))
    derived_status=dict(status='completed',cells=len(counts),selected=len(kept),ai_selected=ai,real_selected=real,
        seed=cfg['seed'],per_cell=selection['per_cell'],selected_by_cell=counts,
        selected_uncompressed_bytes=filtered['audit']['selected_uncompressed_bytes'],
        selected_compressed_bytes=filtered['audit']['selected_compressed_bytes'],
        acquisition='offline_approved_quarantine_subset',network=dict(bytes=0,requests_attempted=0,cell_failures=0),
        source_acquisition_status_sha256=sha256(status_path))
    selected_receipts(filtered,derived_status,[value for _,value in derived_receipts])
    output.mkdir(parents=True); (output/'receipts').mkdir()
    for rel in paths:
        destination=output/rel; destination.parent.mkdir(parents=True,exist_ok=True)
        os.link(source/rel,destination)
    write(output/'selection.json',filtered)
    for name,value in derived_receipts:write(output/'receipts'/name,value)
    report=dict(status='completed',time=time.time(),selected=len(kept),ai_selected=ai,real_selected=real,
        quarantined=len(excluded),parent_samples=len(frozen),parent_preserved=True,network_requests=0,
        holdout_scored=False,images_decoded=0,source_raw_files_modified=False,full_integrity_pending=True,
        source=cfg['source']['path'],source_selection_sha256=sha256(selection_path),
        parent_manifest_sha256=sha256(parent_path),exclusion_sha256=sha256(excluded_path),
        config_sha256=sha256(config_path),script_sha256=sha256(Path(__file__)),
        selected_by_cell=counts,exclusion_ids=sorted(excluded),authorization=cfg['authorization'],
        grouped_metadata_verified=True,group_count=grouping['group_count'])
    write(output/'curation.json',report)
    write(output/'download_status.json',derived_status)  # Completion is last; a partial subset is never called complete.
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--config',type=Path,required=True)
    args=parser.parse_args();report=curate(args.root,args.config)
    print(json.dumps({k:v for k,v in report.items() if k not in ('exclusion_ids','selected_by_cell')}),flush=True)


if __name__=='__main__':main()
