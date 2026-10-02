"""Click/mask contract and persistence across the isolated SAM environment."""
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

from easyhec.segmentation import sam31 as segmentation


@unittest.skipUnless(importlib.util.find_spec('torch') and importlib.util.find_spec('easyhec'),
                     'requires the EasyHeC segmentation environment')
class Sam31Tests(unittest.TestCase):
    def test_click_coordinates_labels_and_reset_between_annotations(self):
        image = np.zeros((40, 80, 3), np.uint8)
        image[3, 4] = [10, 20, 30]
        mask = np.zeros((40, 80), bool)
        mask[5:20, 10:40] = True
        def initialize(**kwargs):
            saved = cv2.cvtColor(cv2.imread(kwargs['resource_path']), cv2.COLOR_BGR2RGB)
            np.testing.assert_array_equal(saved, image)
            return {'cached_frame_outputs': {}}

        model = Mock()
        model.init_state.side_effect = initialize
        model.add_prompt.return_value = (0, {'out_obj_ids': [1], 'out_binary_masks': mask[None]})
        adapter = segmentation.Sam31Segmenter(model)
        try:
            got = adapter(image, [[20, 10, 1], [60, 30, -1]])
            np.testing.assert_array_equal(got, mask)
            prompt = model.add_prompt.call_args.kwargs
            self.assertEqual(prompt['inference_state']['cached_frame_outputs'], {0: {}})
            np.testing.assert_array_equal(prompt['points'], [[.25, .25], [.75, .75]])
            np.testing.assert_array_equal(prompt['point_labels'], [1, 0])
            adapter(image.copy(), [[20, 10, 1]])
            model.init_state.assert_called_once()
            model.reset_state.assert_called_once()
            np.testing.assert_array_equal(model.add_prompt.call_args.kwargs['point_labels'], [1])
            image[0, 0] = [1, 2, 3]
            adapter(image, [[20, 10, 1]])
            self.assertEqual(model.init_state.call_count, 2)
            with self.assertRaisesRegex(ValueError, 'inside'):
                adapter(image, [[80, 10, 1]])
        finally:
            adapter.close()
        self.assertIsNone(adapter.state)
        self.assertFalse(Path(adapter.temp.name).exists())

    def test_empty_output_keeps_full_image_dimensions(self):
        model = Mock()
        model.init_state.return_value = {'cached_frame_outputs': {}}
        model.add_prompt.return_value = (0, dict(out_obj_ids=[], out_binary_masks=np.zeros((0, 40, 80), bool)))
        adapter = segmentation.Sam31Segmenter(model)
        try:
            mask = adapter(np.zeros((40, 80, 3), np.uint8), [[10, 10, -1]])
            self.assertEqual(mask.shape, (40, 80))
            self.assertFalse(mask.any())
        finally:
            adapter.close()

    def test_save_reviewed_masks_skip_existing_and_explicit_overwrite(self):
        # Use the real upstream UI class with only its interactive loop replaced.
        from easyhec.segmentation.interactive import InteractiveSegmentation

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cv2.imwrite(str(root/'image.png'), np.zeros((40, 80, 3), np.uint8))
            old = np.zeros((40, 80), np.uint8)
            old[1:5, 1:5] = 255
            new = np.zeros((40, 80), np.uint8)
            new[10:20, 10:30] = 1
            cv2.imwrite(str(root/'mask.png'), old)
            (root/'dataset.json').write_text(json.dumps(dict(parts=['base'], samples=[
                dict(image='image.png', mask='mask.png')])))
            adapter = Mock(points=[[15, 15, 1]])
            with patch.object(segmentation, 'build_segmenter', return_value=adapter) as build, \
                 patch.object(InteractiveSegmentation, 'get_segmentation', return_value=new[None]):
                segmentation.segment(root)
                build.assert_not_called()
                np.testing.assert_array_equal(cv2.imread(str(root/'mask.png'), 0), old)
                segmentation.segment(root, overwrite=True)
                np.testing.assert_array_equal(cv2.imread(str(root/'mask.png'), 0), new*255)
                metadata = json.loads((root/'mask.json').read_text())
                self.assertEqual(metadata['model'], 'sam3.1')
                self.assertEqual(metadata['clicks_xy_label'], [[15, 15, 1]])
                adapter.close.assert_called_once()

    def test_text_detection_then_refines_clicked_instance_in_same_state(self):
        image = np.zeros((40, 80, 3), np.uint8)
        masks = np.zeros((2, 40, 80), bool)
        masks[0, 5:20, 5:20] = True
        masks[1, 5:20, 50:70] = True
        state = {'cached_frame_outputs': {0: {'text_result': True}}}
        model = Mock()
        model.init_state.return_value = state
        tracker_state = dict(obj_id_to_idx={7: 0, 9: 1}, frames_already_tracked={},
            output_dict={'cond_frame_outputs': {0: {'pred_masks': np.zeros((2, 1, 10, 20))}}})
        model._get_sam2_inference_states_by_obj_ids.return_value = [tracker_state]

        def prompt(**kwargs):
            self.assertIs(kwargs['inference_state'], state)
            if 'text_str' in kwargs:
                self.assertNotIn('points', kwargs)
                return 0, dict(out_obj_ids=[7, 9], out_probs=[.6, .9], out_binary_masks=masks)
            self.assertEqual(kwargs['obj_id'], 7)
            self.assertTrue(model.tracker.model.iter_use_prev_mask_pred)
            self.assertIn(0, tracker_state['user_refined_frames_per_obj'][7])
            self.assertIn(0, tracker_state['frames_already_tracked'])
            self.assertEqual(state['cached_frame_outputs'][0], {'text_result': True})
            np.testing.assert_array_equal(kwargs['point_labels'], [1, 0])
            return 0, dict(out_obj_ids=[7, 9], out_binary_masks=masks)

        model.add_prompt.side_effect = prompt
        adapter = segmentation.Sam31Segmenter(model, text='robot')
        try:
            np.testing.assert_array_equal(adapter(image, []), masks[1])
            self.assertEqual(adapter.object_id, 9)
            # Foreground click selects object 7; background click must not select object 9.
            np.testing.assert_array_equal(adapter(image, [[10, 10, 1], [60, 10, -1]]), masks[0])
            self.assertEqual(adapter.object_id, 7)
            np.testing.assert_array_equal(adapter(image, []), masks[1])
        finally:
            adapter.close()

    def test_text_without_detection_does_not_silently_fall_back_to_points(self):
        model = Mock()
        model.init_state.return_value = {'cached_frame_outputs': {}}
        model.add_prompt.return_value = (0, dict(out_obj_ids=[], out_probs=[],
            out_binary_masks=np.zeros((0, 40, 80), bool)))
        adapter = segmentation.Sam31Segmenter(model, text='robot')
        try:
            mask = adapter(np.zeros((40, 80, 3), np.uint8), [[10, 10, 1]])
            self.assertFalse(mask.any())
            self.assertIsNone(adapter.object_id)
            model.add_prompt.assert_called_once()
        finally:
            adapter.close()

    def test_text_review_regenerates_after_click_before_accepting(self):
        image = np.zeros((40, 80, 3), np.uint8)
        first = np.zeros((40, 80), np.uint8)
        refined = first.copy()
        refined[5:20, 5:20] = 1
        adapter = Mock(text='robot', side_effect=[first, refined])
        callbacks = {}
        keys = iter([ord('t'), ord('t')])

        def key(delay):
            if 'click' in callbacks:
                callbacks.pop('click')(cv2.EVENT_LBUTTONDOWN, 10, 10, 0, None)
            return next(keys)

        with patch.multiple(segmentation.cv2, namedWindow=Mock(), resizeWindow=Mock(),
                setMouseCallback=lambda window, callback: callbacks.update(click=callback),
                imshow=Mock(), waitKey=key, getWindowProperty=Mock(return_value=1),
                destroyAllWindows=Mock()):
            np.testing.assert_array_equal(segmentation.review_text_mask(image, adapter), refined)
        self.assertEqual(adapter.call_count, 2)
        self.assertEqual(adapter.call_args.args[1], [(10, 10, 1)])

    @unittest.skipUnless(os.environ.get('RUN_SAM31_FIXTURE'), 'opt-in real SAM3.1 mask refinement')
    def test_real_text_mask_survives_local_point_correction(self):
        # Fixture JSON supplies image, positive_xy and negative_xy: both points
        # must be within the text-detected object, on different robot parts.
        fixture = json.loads(Path(os.environ['RUN_SAM31_FIXTURE']).read_text())
        image = cv2.cvtColor(cv2.imread(fixture['image']), cv2.COLOR_BGR2RGB)
        adapter = segmentation.build_segmenter(text='robot')
        try:
            initial = adapter(image, []) > 0
            px, py = fixture['positive_xy']
            nx, ny = fixture['negative_xy']
            self.assertTrue(initial[py, px] and initial[ny, nx])
            confirmed = adapter(image, [[px, py, 1]]) > 0
            self.assertGreater((initial & confirmed).sum() / (initial | confirmed).sum(), .9)
            corrected = adapter(image, [[nx, ny, -1]]) > 0
            self.assertFalse(corrected[ny, nx])
            self.assertTrue(corrected[py, px])
            self.assertGreater((initial & corrected).sum() / initial.sum(), .7)
            np.testing.assert_array_equal(adapter(image, []) > 0, initial)
        finally:
            adapter.close()


if __name__ == '__main__':
    unittest.main()
