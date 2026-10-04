"""Offline ablation of saved SAM masks. Writes only a new experiment directory.

Pixel parameters below target 1280x800 images. This is an experiment, not a
production default. Depth must already be aligned with the rectified RGB.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np


def kernel(radius):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1,) * 2)


def components(mask):
    return cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)


def small_holes(mask, area=128):
    _, labels, stats, _ = components(~mask)
    ids = np.flatnonzero(stats[:, cv2.CC_STAT_AREA] <= area)
    exterior = np.unique(np.r_[labels[0], labels[-1], labels[:, 0], labels[:, -1]])
    ids = np.setdiff1d(ids, np.r_[0, exterior])
    return np.isin(labels, ids)


def cleanup(mask, radius=2, hole_area=128):
    # Fill enclosed pinholes, not all background gaps or cable loops.
    result = mask | small_holes(mask, hole_area)
    result = cv2.morphologyEx(result.astype(np.uint8), cv2.MORPH_OPEN, kernel(radius))
    # Small smoothing kernel; avoid global closing that joins fingers.
    result = cv2.GaussianBlur(result.astype(np.float32), (0, 0), .8) >= .5
    _, labels, stats, _ = components(result)
    ids = np.flatnonzero(stats[:, cv2.CC_STAT_AREA] >= 32)
    return np.isin(labels, ids[ids != 0])


def depth_rejection(mask, depth_m):
    valid = np.isfinite(depth_m) & (depth_m > 0)
    core = cv2.erode(mask.astype(np.uint8), kernel(10)).astype(bool) & valid
    if core.sum() < 100:
        raise ValueError('Too few valid core depth pixels')
    # A deliberately broad scene-derived foreground band; do not reject unknowns.
    lo, hi = np.percentile(depth_m[core], [2, 95]) + [-.1, .1]
    bad = valid & ((depth_m < lo) | (depth_m > hi))
    # Only trust locally supported outliers, not isolated stereo failures.
    supported = cv2.boxFilter(bad.astype(np.float32), -1, (5, 5)) >= .8
    boundary = ~cv2.erode(mask.astype(np.uint8), kernel(8)).astype(bool)
    return bad, supported & boundary, (float(lo), float(hi))


def grabcut(image, mask, reject=None):
    seeds = np.full(mask.shape, cv2.GC_BGD, np.uint8)
    seeds[cv2.dilate(mask.astype(np.uint8), kernel(5)) > 0] = cv2.GC_PR_BGD
    seeds[mask] = cv2.GC_PR_FGD
    seeds[cv2.erode(mask.astype(np.uint8), kernel(5)) > 0] = cv2.GC_FGD
    if reject is not None:
        seeds[reject & (seeds != cv2.GC_FGD)] = cv2.GC_BGD
    cv2.setRNGSeed(0)
    cv2.grabCut(image, seeds, None, np.zeros((1, 65)), np.zeros((1, 65)), 3,
                cv2.GC_INIT_WITH_MASK)
    return (seeds == cv2.GC_FGD) | (seeds == cv2.GC_PR_FGD)


def overlay(image, mask):
    result = image.copy()
    result[mask] = (result[mask] * .55 + np.array([50, 220, 50]) * .45).astype(np.uint8)
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(result, contours, -1, (40, 255, 40), 1)
    return result


def label(image, text):
    result = image.copy()
    cv2.rectangle(result, (0, 0), (result.shape[1], 31), (25, 25, 25), -1)
    cv2.putText(result, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 1)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    doc = json.loads((args.dataset / 'dataset.json').read_text())
    meta = doc['camera_metadata']
    K = doc['intrinsics']['camera_matrix']
    if (not np.allclose(meta['depth_K'], K)
            or not np.allclose(meta['depth_to_color_R'], np.eye(3).ravel())
            or not np.allclose(meta['depth_to_color_t'], 0)
            or not np.allclose(meta['dist_coeffs'], 0)):
        raise ValueError('This experiment requires depth aligned to rectified RGB')
    metrics, manifest, cards = [], [], []
    for index, sample in enumerate(doc['samples']):
        paths = {k: args.dataset / sample[k] for k in ['image', 'mask', 'depth']}
        manifest.append({k: {'path': str(p.resolve()), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
                         for k, p in paths.items()})
        image = cv2.imread(str(paths['image']))
        mask = cv2.imread(str(paths['mask']), 0) > 0
        depth = cv2.imread(str(paths['depth']), cv2.IMREAD_UNCHANGED) * sample['depth_scale_m']
        if image.shape[:2] != mask.shape or depth.shape != mask.shape:
            raise ValueError('Image/mask/depth shape mismatch')
        bad, conservative_bad, band = depth_rejection(mask, depth)
        target = args.output / f'{index:02d}'
        target.mkdir()
        variants = {'original': lambda: mask,
                    'holes128': lambda: mask | small_holes(mask, 128),
                    'clean_r2': lambda: cleanup(mask, 2),
                    'clean_r4': lambda: cleanup(mask, 4),
                    'clean_r6': lambda: cleanup(mask, 6),
                    'depth_naive': lambda: mask & (depth > 0) & ~bad,
                    'depth_keep_unknown': lambda: mask & ~bad,
                    'depth_conservative': lambda: cleanup(mask & ~conservative_bad, 2),
                    'rgb_grabcut': lambda: cleanup(grabcut(image, mask), 2),
                    'rgb_depth_grabcut': lambda: cleanup(grabcut(image, mask, conservative_bad), 2)}
        panels = []
        for name, run in variants.items():
            start = time.perf_counter()
            result = run()
            elapsed = time.perf_counter() - start
            cv2.imwrite(str(target / f'{name}.png'), result.astype(np.uint8) * 255)
            vis = overlay(image, result)
            cv2.imwrite(str(target / f'{name}_overlay.jpg'), vis)
            change = image.copy()
            change[mask & ~result] = (0, 50, 255)  # removed: red
            change[~mask & result] = (255, 220, 0)  # added: cyan
            cv2.imwrite(str(target / f'{name}_changes.jpg'), change)
            panels.append(label(cv2.resize(vis, (640, 400)), name))
            n, _, _, _ = components(result)
            nh, _, _, _ = components(small_holes(result, 128))
            metrics.append(dict(frame=index, method=name, area=int(result.sum()),
                                added=int((result & ~mask).sum()), removed=int((mask & ~result).sum()),
                                components=n-1, small_holes=nh-1,
                                small_hole_pixels=int(small_holes(result, 128).sum()),
                                valid_depth_fraction=float((depth[mask] > 0).mean()),
                                band_low_m=band[0], band_high_m=band[1], seconds=elapsed))
            print(index, name, metrics[-1]['added'], metrics[-1]['removed'], flush=True)
        cv2.imwrite(str(target / 'comparison.jpg'), np.vstack([np.hstack(panels[i:i+2]) for i in range(0, len(panels), 2)]))
        # Fixed review crops for the eight-frame OAK fixture; coordinates in pixels.
        crops = {0: (710, 460, 1110, 770), 1: (300, 315, 795, 525),
                 2: (40, 310, 360, 630), 3: (565, 440, 880, 760),
                 4: (800, 380, 1240, 640), 5: (600, 300, 950, 550),
                 6: (580, 440, 930, 790), 7: (620, 440, 990, 790)}
        if image.shape[:2] == (800, 1280) and index in crops:
            x1, y1, x2, y2 = crops[index]
            detail = []
            for pair in [('original', 'holes128'), ('clean_r2', 'clean_r6'),
                         ('rgb_grabcut', 'rgb_depth_grabcut')]:
                tiles = [label(cv2.resize(cv2.imread(str(target / f'{name}_overlay.jpg'))
                         [y1:y2, x1:x2], None, fx=1.5, fy=1.5), name) for name in pair]
                detail.append(np.hstack(tiles))
            cv2.imwrite(str(target / 'detail.jpg'), np.vstack(detail))
        cv2.imwrite(str(target / 'rgb.jpg'), image)
        cards.append(f'<h2>Frame {index}: {paths["image"].stem}</h2><p><a href="{index:02d}/rgb.jpg">RGB</a> | '
                     + ' | '.join(f'<a href="{index:02d}/{n}_overlay.jpg">{n}</a> '
                                  f'(<a href="{index:02d}/{n}_changes.jpg">changes</a>)' for n in variants)
                     + f'</p><img loading="lazy" src="{index:02d}/comparison.jpg">')
    with (args.output / 'metrics.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=metrics[0].keys()); writer.writeheader(); writer.writerows(metrics)
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    (args.output / 'index.html').write_text('<!doctype html><meta charset="utf-8"><title>OAK mask experiments</title>'
        '<style>body{font-family:system-ui;background:#202020;color:#eee;margin:24px}a{color:#8cf}'
        'img{max-width:100%}p{max-width:1200px}</style><h1>OAK mask experiments</h1>'
        '<p>Eight saved SAM3.1 masks; source files untouched. Green=mask. Change views: red=removed, cyan=added. '
        'No ground truth: pixel counts measure edits, not accuracy. Parameters are in refine_saved_masks.py.</p>'
        + ''.join(cards))


if __name__ == '__main__':
    main()
