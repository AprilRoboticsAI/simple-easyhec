"""Mask topology, deterministic manual corrections and review/save contracts."""
import json
import tempfile
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

from easyhec.segmentation.mask_editor import MaskEditor, fill_small_holes, remove_small_islands, review_mask
from easyhec.segmentation import sam31


class CleanupTests(unittest.TestCase):
    def test_islands_preserve_large_disconnected_parts_and_positive_seed(self):
        mask = np.zeros((80, 120), bool)
        mask[10:30, 10:30] = True
        mask[40:60, 80:100] = True  # valid disconnected hand, not largest-only
        mask[2:4, 2:4] = True
        mask[70:72, 110:112] = True
        result = remove_small_islands(mask, foreground_points=[(110, 70), (0, 0)])
        self.assertFalse(result[2, 2])
        self.assertTrue(result[70, 110])
        self.assertTrue(result[15, 15] and result[45, 85])
        self.assertFalse(result[0, 0])

    def test_holes_preserve_exterior_large_gaps_and_border_pockets(self):
        mask = np.ones((80, 80), bool)
        mask[20:23, 20:23] = False
        mask[30:50, 30:50] = False
        mask[:3, :3] = False
        result = fill_small_holes(mask)
        self.assertTrue(result[21, 21])
        self.assertFalse(result[40, 40])
        self.assertFalse(result[1, 1])
        np.testing.assert_array_equal(fill_small_holes(mask.astype(np.uint8)*255), result)

    def test_opening_cuts_thin_spur_without_collapsing_main_body(self):
        mask = np.zeros((80, 80), np.uint8)
        mask[20:60, 20:60] = 1
        mask[38:40, 60:75] = 1
        editor = MaskEditor(mask)
        result = editor.result()
        self.assertFalse(result[38, 70])
        self.assertTrue(result[40, 40])
        self.assertGreater(result.sum(), 1500)
        np.testing.assert_array_equal(editor.raw, mask > 0)

    def test_brush_overrides_cleanup_and_survives_new_sam_prediction(self):
        mask = np.ones((80, 80), np.uint8)
        editor = MaskEditor(mask)
        editor.begin_stroke(20, 20, 0, 2)
        editor.extend_stroke(50, 20)
        self.assertFalse(editor.result()[20, 30])
        editor.regenerate(mask, object_id=9)
        self.assertFalse(editor.result()[20, 30])
        editor.begin_stroke(30, 20, 1, 1)
        self.assertTrue(editor.result()[20, 30])
        editor.undo()
        self.assertFalse(editor.result()[20, 30])
        editor.undo()  # regeneration, retaining the previous manual erasure
        self.assertIsNone(editor.object_id)
        self.assertFalse(editor.result()[20, 30])

    def test_restored_island_survives_island_removal_and_stroke_replay(self):
        editor = MaskEditor(np.zeros((80, 80), np.uint8))
        editor.begin_stroke(0, 0, 1, 2)
        editor.extend_stroke(10, 0)
        editor.begin_stroke(6, 0, 0, 1)
        meta = json.loads(json.dumps(editor.metadata()))
        replay = MaskEditor(editor.raw)
        for stroke in meta['strokes']:
            replay.begin_stroke(*stroke['xy'][0], stroke['value'], stroke['radius'])
            for point in stroke['xy'][1:]:
                replay.extend_stroke(*point)
        np.testing.assert_array_equal(editor.result(), replay.result())
        self.assertTrue(editor.result()[0, 0])
        self.assertFalse(editor.result()[0, 6])

    def test_reset_and_undo_restore_prompts_mask_and_strokes(self):
        editor = MaskEditor(np.ones((40, 40), np.uint8), object_id=3)
        editor.add_point(10, 10, 1)
        editor.regenerate(editor.raw, object_id=7)
        editor.begin_stroke(15, 15, 0, 2)
        before = editor.result()
        editor.reset(np.zeros((40, 40), np.uint8), pending=True)
        self.assertFalse(editor.result().any())
        self.assertFalse(editor.strokes)
        editor.undo()
        np.testing.assert_array_equal(editor.result(), before)
        self.assertEqual(editor.object_id, 7)
        self.assertEqual(editor.points, [(10, 10, 1)])
        self.assertTrue(editor.metadata()['cleanup']['opening'])
        self.assertFalse(editor.pending)


class ReviewTests(unittest.TestCase):
    def run_review(self, adapter, steps):
        callback = None
        steps = iter(steps)
        def register(window, handler):
            nonlocal callback
            callback = handler
        def key(delay):
            events, keycode = next(steps)
            for event, x, y, flags in events:
                callback(event, x, y, flags, None)
            return keycode
        with patch.multiple(cv2, namedWindow=Mock(), resizeWindow=Mock(),
                setMouseCallback=register, imshow=Mock(), waitKey=key,
                getWindowProperty=Mock(return_value=1), destroyWindow=Mock()):
            return review_mask(np.zeros((120, 160, 3), np.uint8), adapter)

    def test_drag_erase_restore_then_regenerate_preserves_edits_and_metadata(self):
        mask = np.ones((120, 160), np.uint8)
        adapter = Mock(text='robot', object_id=7, side_effect=[mask, mask])
        down = cv2.EVENT_LBUTTONDOWN
        result = self.run_review(adapter, [([], ord('b')),
            ([(down, 30, 80, 0), (cv2.EVENT_MOUSEMOVE, 70, 80, cv2.EVENT_FLAG_LBUTTON),
              (cv2.EVENT_LBUTTONUP, 70, 80, 0)], 255),
            ([(cv2.EVENT_RBUTTONDOWN, 50, 80, 0), (cv2.EVENT_RBUTTONUP, 50, 80, 0)], ord('e')),
            ([(down, 100, 100, 0)], ord('t')), ([], ord('t'))])
        self.assertFalse(result[80, 20])
        self.assertTrue(result[80, 50])
        self.assertEqual(adapter.call_count, 2)
        self.assertEqual(adapter.points, [(100, 100, 1)])
        self.assertEqual(len(adapter.review_metadata['strokes']), 2)
        self.assertEqual(adapter.review_metadata['strokes'][0]['radius'], 24)
        self.assertTrue(adapter.review_metadata['cleanup']['holes'])

    def test_point_mode_requires_generation_before_accepting(self):
        mask = np.zeros((120, 160), np.uint8); mask[30:90, 30:90] = 1
        adapter = Mock(text=None, object_id=1, return_value=mask)
        result = self.run_review(adapter, [([], ord('t')),
            ([(cv2.EVENT_LBUTTONDOWN, 40, 40, 0)], ord('t')), ([], ord('t'))])
        np.testing.assert_array_equal(result, MaskEditor(mask).result())
        adapter.assert_called_once()

    def test_undo_stroke_and_original_comparison_cannot_accept_unseen_edits(self):
        mask = np.ones((120, 160), np.uint8)
        adapter = Mock(text='robot', object_id=1, return_value=mask)
        result = self.run_review(adapter, [([], ord('b')),
            ([(cv2.EVENT_LBUTTONDOWN, 30, 80, 0), (cv2.EVENT_LBUTTONUP, 30, 80, 0)], ord('u')),
            ([], ord('v')), ([], ord('t')), ([], ord('t'))])
        np.testing.assert_array_equal(result, MaskEditor(mask).result())
        self.assertEqual(adapter.review_metadata['strokes'], [])

    def test_empty_mask_not_accepted_and_cancellation_does_not_save(self):
        adapter = Mock(text='robot', object_id=1, return_value=np.zeros((120, 160), np.uint8))
        with self.assertRaisesRegex(ValueError, 'cancelled'):
            self.run_review(adapter, [([], ord('t')), ([], 27)])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cv2.imwrite(str(root/'image.png'), np.zeros((120, 160, 3), np.uint8))
            old = np.ones((120, 160), np.uint8)*255
            cv2.imwrite(str(root/'mask.png'), old)
            (root/'mask.json').write_text('{"original": true}')
            (root/'dataset.json').write_text(json.dumps(dict(parts=['base'], samples=[dict(image='image.png', mask='mask.png')])))
            with patch.object(sam31, 'build_segmenter', return_value=adapter), \
                 patch.object(sam31, 'review_text_mask', side_effect=ValueError('cancelled')):
                with self.assertRaisesRegex(ValueError, 'cancelled'):
                    sam31.segment(root, overwrite=True, text='robot')
            np.testing.assert_array_equal(cv2.imread(str(root/'mask.png'), 0), old)
            self.assertEqual(json.loads((root/'mask.json').read_text()), {'original': True})
            adapter.close.assert_called_once()


if __name__ == '__main__':
    unittest.main()
