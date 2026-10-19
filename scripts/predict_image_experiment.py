"""Single/batch offline inference from a saved research checkpoint, never a cloud API."""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from PIL import Image
from research.image_attribution.models import load_checkpoint, resnet_transform
from research.image_attribution.protocol import sha256


def predict(model_path, images, device='cpu', batch_size=8):
    try:
        model, checkpoint = load_checkpoint(model_path, device)
        model_hash = sha256(model_path)
    except (OSError, ValueError, KeyError, RuntimeError, pickle.UnpicklingError, EOFError) as exc:
        return [dict(path=str(path), status='unavailable', score=None, top_k=[], error=str(exc)) for path in images]
    transform = resnet_transform()
    results = [None] * len(images)
    for start in range(0, len(images), batch_size):
        inputs, indices = [], []
        for index in range(start, min(start + batch_size, len(images))):
            try:
                with Image.open(images[index]) as image:
                    inputs.append(transform(image.convert('RGB')))
                indices.append(index)
            except (OSError, ValueError) as exc:
                results[index] = dict(path=str(images[index]), status='unsupported', score=None, top_k=[], error=str(exc))
        if not inputs:
            continue
        try:
            with torch.inference_mode():
                probs = model(torch.stack(inputs).to(device)).softmax(1).cpu().tolist()
            for index, scores in zip(indices, probs):
                result = dict(path=str(images[index]), status='ok', task=checkpoint['task'],
                              model_sha256=model_hash, classes=checkpoint['classes'], scores=scores)
                if checkpoint['task'] == 'detection':
                    result.update(score=scores[1], threshold=.5, label='ai' if scores[1] >= .5 else 'real',
                                  note='Research score; fixed 0.5 operating point, not calibrated probability')
                else:
                    result.update(top_k=[dict(model=checkpoint['classes'][i], probability=scores[i])
                        for i in sorted(range(len(scores)), key=lambda i:scores[i], reverse=True)[:5]],
                        confidence=max(scores), unknown_probability=None,
                        note='Closed candidate set only; unknown rejection and truth detection not provided')
                results[index] = result
        except RuntimeError as exc:
            for index in indices:
                results[index] = dict(path=str(images[index]),status='unavailable',score=None,top_k=[],error=str(exc))
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--images', type=Path, nargs='+', required=True)
    p.add_argument('--device', choices=['cpu','mps','cuda'], default='cpu')
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.batch_size < 1:
        p.error('batch size must be positive')
    torch.set_num_threads(4)
    results = predict(a.model,a.images,a.device,a.batch_size)
    with a.output.open('x') as stream:
        stream.write(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in results))
    if any(r['status']!='ok' for r in results):
        raise SystemExit(1)


if __name__=='__main__':
    main()
