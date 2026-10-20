"""Evaluate image-only cleanup; writes a new output directory, never source masks."""
import argparse
import json
from pathlib import Path
import time

import cv2
import numpy as np

from easyhec.segmentation.mask_editor import MaskEditor
from easyhec.segmentation.sam31 import build_segmenter


def panel(image, mask, title, color=(40, 230, 40)):
    display = image.copy()
    display[mask] = .55 * display[mask] + .45 * np.array(color)
    cv2.rectangle(display, (0, 0), (display.shape[1], 36), (20, 20, 20), -1)
    cv2.putText(display, title, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, .65, (255, 255, 255), 1)
    return display


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    doc = json.loads((args.dataset/'dataset.json').read_text())
    prompts = dict(include=['vertical metal support', 'black vertical post'] if 'base' in doc['parts'] else [], exclude=['cable'], protect=['robot hand'], recover_hands=bool(doc.get('hands')))
    segmenter = build_segmenter(text='robot')
    rows, cards = [], []
    try:
        for i, sample in enumerate(doc['samples']):
            started = time.monotonic()
            bgr = cv2.imread(str(args.dataset/sample['image']))
            if bgr is None:
                raise ValueError(f'Cannot read {sample["image"]}')
            raw = segmenter(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), [])
            cleanup = dict(prompts=prompts, hands=segmenter.detect_objects(prompts['protect']), additions=segmenter.detect_objects(prompts['include']),
                           exclusions=segmenter.detect_exclusions(prompts['exclude'], raw))
            baseline = MaskEditor(raw).result()
            editor = MaskEditor(raw, cleanup=cleanup)
            result = editor.result()
            added, removed = result & ~baseline, baseline & ~result
            row = dict(index=i, image=sample['image'], seconds=time.monotonic()-started,
                       cleanup=editor.metadata()['image_cleanup'])
            rows.append(row)
            for name, mask in [('baseline', baseline), ('automatic', result), ('added', added), ('removed', removed)]:
                if not cv2.imwrite(str(args.output/f'{i:03d}-{name}.png'),mask.astype(np.uint8)*255):
                    raise OSError('Could not write comparison mask')
            before = panel(bgr, baseline, 'Before: robot prompt + existing cleanup')
            after = panel(bgr, result, 'After: support added, cables subtracted (image only)')
            changes = panel(bgr, added, 'Changes: cyan added, magenta removed', (255,220,40))
            changes[removed] = .4*changes[removed]+.6*np.array([230,40,230])
            original = panel(bgr, np.zeros_like(result), 'Original image')
            sheet = np.concatenate([np.concatenate([before,after],1),np.concatenate([original,changes],1)],0)
            cv2.imwrite(str(args.output/f'{i:03d}-comparison.jpg'),sheet)
            cards.append(cv2.resize(after,(480,300)))
            (args.output/'results.json').write_text(json.dumps(rows,indent=2)+'\n')
            print(json.dumps(row),flush=True)
    finally:
        segmenter.close()
    while len(cards)%3:
        cards.append(np.zeros_like(cards[0]))
    cv2.imwrite(str(args.output/'contact-sheet.jpg'),np.concatenate([np.concatenate(cards[i:i+3],1) for i in range(0,len(cards),3)],0))
    html = '<!doctype html><meta charset="utf-8"><title>Image-only mask cleanup</title><style>body{background:#181818;color:white;font:16px sans-serif}img{max-width:100%}</style><h1>Image-only mask cleanup</h1><p>No camera estimate or robot reprojection. Green: resulting mask. Cyan: additions. Magenta: removals.</p>'
    for row in rows:
        html += f'<h2>Frame {row["index"]}</h2><img loading="lazy" src="{row["index"]:03d}-comparison.jpg">'
    (args.output/'index.html').write_text(html)


if __name__ == '__main__':
    main()
