"""SAM 3.1 backend for EasyHeC's click UI; run in environments/sam31.

The isolated environment pins the dependencies required by upstream SAM3.
The predictor treats each rectified capture as an independent still image.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import io
import json
import logging
from pathlib import Path
import tempfile

import cv2
import numpy as np

SAM3_REVISION = '2345a4ad109ac29c569da749c91d84f10dc08c40'
SAM31_CHECKPOINT_REVISION = 'daa63191845a41281374e725f4c9e51c7a824460'


class Sam31Segmenter:
    """Translate EasyHeC pixel clicks (+1/-1) to SAM 3.1 point prompts (1/0)."""

    def __init__(self, model, text=None):
        self.model = model
        self.text = text
        self.object_id = None
        self.state = None
        self.image = None
        self.points = None
        self.temp = tempfile.TemporaryDirectory(prefix='easyhec-sam31-')

    def __call__(self, image, clicked_points):
        import torch

        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            return self._segment(image, clicked_points)

    def _segment(self, image, clicked_points):
        import torch

        points = np.asarray(clicked_points)
        if points.size == 0 and self.text:
            points = np.empty((0, 3))
        if (points.ndim != 2 or points.shape[1] != 3 or (not len(points) and not self.text)
                or not np.isfinite(points).all()
                or not np.isin(points[:, 2], [-1, 1]).all()):
            raise ValueError('Expected nonempty finite (x, y, +1/-1) clicks')
        height, width = image.shape[:2]
        if ((points[:, :2] < 0).any() or (points[:, 0] >= width).any()
                or (points[:, 1] >= height).any()):
            raise ValueError('Clicks must lie inside the rectified image')
        # Upstream UI passes image copies; compare pixels to reuse the loaded frame.
        if self.image is None or not np.array_equal(self.image, image):
            self.state = None
            path = Path(self.temp.name) / 'image.png'
            if not cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR)):
                raise OSError('Could not save temporary SAM input')
            # The pinned upstream session wrapper passes offload_state_to_cpu,
            # which its multiplex model does not accept. Use the model API for
            # our single-image state; video session management is unnecessary.
            self.state = self.model.init_state(resource_path=str(path), async_loading_frames=False)
            self.image = image.copy()
        else:
            # r/e in the UI must not retain an earlier mask or removed prompts.
            self.model.reset_state(self.state)
        self.points = points.tolist()
        self.object_id = 1
        if self.text:
            # SAM3 disallows text and points in one call. First detect instances,
            # then refine the chosen instance in that same inference state.
            _, output = self.model.add_prompt(
                inference_state=self.state, frame_idx=0, text_str=self.text)
            masks, ids = self._masks(output, height, width)
            if not len(ids):
                self.object_id = None
                print(f'No object found for {self.text!r}; try a different text prompt.')
                return np.zeros((height, width), np.uint8)
            scores = np.asarray(output['out_probs'])
            positive = points[points[:, 2] > 0, :2].astype(int)
            hits = masks[:, positive[:, 1], positive[:, 0]].sum(axis=1)
            # A single mask is calibrated. Positive clicks select among detected
            # instances; confidence breaks ties (including the text-only case).
            selected = max(range(len(ids)), key=lambda i: (hits[i], scores[i]))
            self.object_id = int(ids[selected])
            print(f'Text {self.text!r}: {len(ids)} candidate(s), refining object {self.object_id}')
            if not len(points):
                return masks[selected].astype(np.uint8)
            self._use_detected_mask_for_refinement()
        else:
            # Upstream drops a first point-only result unless its frame cache exists.
            self.state['cached_frame_outputs'][0] = {}
        _, output = self.model.add_prompt(
            inference_state=self.state, frame_idx=0, obj_id=self.object_id,
            points=torch.tensor(points[:, :2] / [width, height], dtype=torch.float32),
            point_labels=torch.tensor(points[:, 2] > 0, dtype=torch.int32),
            rel_coordinates=True, clear_old_points=True)
        masks, ids = self._masks(output, height, width)
        # Empty output is valid (e.g. negative-only prompts); UI allows editing.
        return np.any(masks[ids == self.object_id], axis=0).astype(np.uint8)

    def _use_detected_mask_for_refinement(self):
        """Keep the text mask as the prior for the first click on this still.

        Pinned SAM3.1's demo treats a first click as fresh segmentation. Its
        subsequent-refinement path accepts previous low-resolution mask logits.
        The detector has already initialized those logits in the tracker; mark
        this frame as initialized for interaction and enable that existing path.
        These state fields are specific to SAM3_REVISION and covered by the
        opt-in real-model regression test.
        """
        states = self.model._get_sam2_inference_states_by_obj_ids(self.state, [self.object_id])
        if len(states) != 1:
            raise ValueError('Cannot find the detected object state for mask refinement')
        state = states[0]
        # add_prompt extracts a detected object from a multi-object batch before
        # refining it. That extraction creates a fresh state and drops the
        # interaction flags below. Extract FIRST, then initialize the state that
        # will actually receive the click (also preserves the object's logits).
        if len(state['obj_ids']) > 1 and not self.model.tracker.per_obj_inference:
            rank = self.model._get_gpu_id_by_obj_id(self.state, self.object_id)
            if rank != self.model.rank:
                raise ValueError('Still-image mask refinement requires the object on the local GPU')
            self.model._extract_object_to_singleton_state(self.state, self.object_id, rank)
            states = self.model._get_sam2_inference_states_by_obj_ids(self.state, [self.object_id])
            if len(states) != 1 or states[0]['obj_ids'] != [self.object_id]:
                raise ValueError('Cannot isolate the detected object for mask refinement')
            state = states[0]
        index = state['obj_id_to_idx'][self.object_id]
        previous = state['output_dict']['cond_frame_outputs'].get(0, {}).get('pred_masks')
        if previous is None or previous.ndim != 4 or previous.shape[0] <= index:
            raise ValueError('The detected object has no mask logits for refinement')
        self.model.tracker.model.iter_use_prev_mask_pred = True
        state['frames_already_tracked'][0] = {'reverse': False}
        state.setdefault('user_refined_frames_per_obj', {}).setdefault(self.object_id, set()).add(0)

    @staticmethod
    def _masks(output, height, width):
        masks = np.asarray(output['out_binary_masks'])
        ids = np.asarray(output['out_obj_ids'])
        if masks.shape != (len(ids), height, width):
            raise ValueError(f'SAM returned unexpected mask shape: {masks.shape}')
        return masks, ids

    def close(self):
        self.state = None
        self.image = None
        self.temp.cleanup()


def build_segmenter(checkpoint=None, text=None):
    import torch
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import HfHubHTTPError, LocalEntryNotFoundError

    if not torch.cuda.is_available():
        raise ValueError('SAM 3.1 requires a CUDA GPU')
    if checkpoint is None:
        try:
            download = dict(repo_id='facebook/sam3.1', filename='sam3.1_multiplex.pt',
                            revision=SAM31_CHECKPOINT_REVISION)
            try:
                checkpoint = hf_hub_download(**download, local_files_only=True)
            except LocalEntryNotFoundError:
                checkpoint = hf_hub_download(**download)
        except HfHubHTTPError as exc:
            raise ValueError('Could not download SAM 3.1 weights. Request access at '
                'https://huggingface.co/facebook/sam3.1 and run '
                '`uv run --project environments/sam31 hf auth login`, '
                'or supply --checkpoint /path/sam3.1_multiplex.pt') from exc
    if not Path(checkpoint).is_file():
        raise ValueError(f'Checkpoint does not exist: {checkpoint}')
    from sam3.model_builder import build_sam3_predictor

    print('Loading SAM 3.1 checkpoint...', flush=True)
    # Upstream prints thousands of mismatched keys while initializing an
    # intermediate tracker, then loads the complete model correctly afterward.
    # Keep that diagnostic available at DEBUG; reject final-model mismatches.
    diagnostic = io.StringIO()
    with redirect_stdout(diagnostic):
        predictor = build_sam3_predictor(
            checkpoint_path=str(checkpoint), version='sam3.1',
            compile=False, warm_up=False, use_fa3=False, use_rope_real=False,
            async_loading_frames=False)
    logging.getLogger(__name__).debug('SAM3 initialization: %s', diagnostic.getvalue())
    # Upstream opens a process-wide autocast context. Scope ours to each call.
    predictor.bf16_context.__exit__(None, None, None)
    if 'Missing keys (' in diagnostic.getvalue() or 'Unexpected keys (' in diagnostic.getvalue():
        raise ValueError('SAM 3.1 checkpoint does not match the pinned full model')
    return Sam31Segmenter(predictor.model, text=text)


def review_text_mask(image, segmenter):
    """Shared cleanup/brush review for text- and point-initialized SAM masks."""
    from easyhec.segmentation.mask_editor import review_mask

    return review_mask(image, segmenter)


def segment(dataset, checkpoint=None, overwrite=False, text=None):
    doc = json.loads((dataset / 'dataset.json').read_text())
    samples = [s for s in doc['samples'] if overwrite or not (dataset / s['mask']).exists()]
    if not samples:
        print('All samples already have masks; use --overwrite to review replacements')
        return
    segmenter = build_segmenter(checkpoint, text=text)
    try:
        print('SAM 3.1: select only', ', '.join(doc['parts']),
              '(exclude objects outside the calibration geometry).')
        for index, sample in enumerate(samples):
            image = cv2.imread(str(dataset / sample['image']))
            if image is None:
                raise ValueError(f'Cannot read {sample["image"]}')
            print(f'Image {index + 1}/{len(samples)}: {sample["image"]}')
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mask = review_text_mask(rgb, segmenter)
            path = dataset / sample['mask']
            path.parent.mkdir(parents=True, exist_ok=True)
            # Save each reviewed frame immediately; an interrupted session can resume.
            temporary = path.with_name(path.stem + '.tmp.png')
            if not cv2.imwrite(str(temporary), (mask > 0).astype(np.uint8) * 255):
                raise OSError(f'Could not save {path}')
            temporary.replace(path)
            provenance = dict(model='sam3.1', source_revision=SAM3_REVISION,
                checkpoint=str(checkpoint) if checkpoint else 'facebook/sam3.1/sam3.1_multiplex.pt',
                checkpoint_revision=None if checkpoint else SAM31_CHECKPOINT_REVISION,
                text_prompt=text, object_id=segmenter.object_id if text else 1,
                refinement='text_mask_and_points' if text else 'points_only',
                image=sample['image'], clicks_xy_label=segmenter.points,
                mask_review=segmenter.review_metadata)
            path.with_suffix('.json').write_text(json.dumps(provenance, indent=2) + '\n')
            print(f'Saved {path}')
    finally:
        segmenter.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--text', help='Find an object by text, then refine it with clicks')
    args = parser.parse_args()
    if args.text is not None and not args.text.strip():
        parser.error('--text must be nonempty')
    try:
        segment(args.dataset, args.checkpoint, args.overwrite, args.text)
    except KeyboardInterrupt:
        parser.exit(130, 'Annotation cancelled; previously accepted masks are saved.\n')
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(1, f'easyhec SAM 3.1: {exc}\n')


if __name__ == '__main__':
    main()
