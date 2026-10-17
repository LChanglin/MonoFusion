"""Evaluate dense NPZ optical flow against KITTI 16-bit ground-truth PNGs.
Local validation only: not a KITTI test-server result, MiD metric, or TTC metric.
"""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np


def evaluate(predictions, ground_truth, ids=None):
    files = ([ground_truth / f'{scene}_10.png' for scene in ids] if ids else sorted(ground_truth.glob('*_10.png')))
    if not files:
        raise ValueError('No ground-truth *_10.png files found')
    rows = []
    total_error = 0.0
    total_bad = total_valid = 0
    for file in files:
        scene = file.stem.removesuffix('_10')
        raw = cv2.imread(str(file), cv2.IMREAD_UNCHANGED)
        if raw is None or raw.dtype != np.uint16 or raw.ndim != 3 or raw.shape[2] != 3:
            raise ValueError(f'Expected a KITTI uint16 three-channel PNG: {file}')
        raw = raw[..., ::-1]
        gt = (raw[..., :2].astype(np.float64) - 32768.0) / 64.0
        valid = raw[..., 2] > 0
        with np.load(predictions / f'{scene}.npz', allow_pickle=False) as archive:
            flow = archive['flow'].astype(np.float64)
        if flow.shape != gt.shape or not np.isfinite(flow).all():
            raise ValueError(f'Prediction shape or finite-value failure: {scene}')
        count = int(valid.sum())
        if not count:
            raise ValueError(f'No valid ground-truth pixels: {scene}')
        error = np.linalg.norm(flow - gt, axis=-1)[valid]
        magnitude = np.linalg.norm(gt, axis=-1)[valid]
        # KITTI's outlier test requires BOTH absolute and relative error thresholds.
        bad = (error > 3.0) & (error > 0.05 * magnitude)
        rows.append({'scene': scene, 'valid_pixels': count, 'epe': float(error.mean()), 'fl_all_percent': float(bad.mean() * 100)})
        total_valid += count
        total_error += float(error.sum())
        total_bad += int(bad.sum())
    return {'evaluation': 'local KITTI flow validation; dense predictions; all valid GT pixels',
            'ground_truth': str(ground_truth.resolve()), 'predictions': str(predictions.resolve()),
            'images': len(rows), 'valid_pixels': total_valid,
            'pixel_weighted_epe': total_error / total_valid,
            'pixel_weighted_fl_all_percent': 100 * total_bad / total_valid,
            'image_mean_epe': float(np.mean([r['epe'] for r in rows])),
            'image_mean_fl_all_percent': float(np.mean([r['fl_all_percent'] for r in rows])), 'per_image': rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--ground-truth', type=Path, required=True)
    parser.add_argument('--ids', nargs='+', help='Explicit held-out scene IDs, e.g. 000000 000005; default: every GT image')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.ids and (len(set(args.ids)) != len(args.ids) or any(len(x) != 6 or not x.isdigit() for x in args.ids)):
        parser.error('--ids must contain unique six-digit scene IDs')
    result = evaluate(args.predictions, args.ground_truth, args.ids)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'per_image'}, indent=2))


if __name__ == '__main__':
    main()
