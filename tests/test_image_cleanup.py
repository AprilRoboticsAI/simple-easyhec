"""Image-only composition and review edits without camera or geometry inputs."""
import unittest
from unittest.mock import Mock

import numpy as np

from easyhec.segmentation.image_cleanup import (compose_mask, repair_silhouette,
                                               cable_loop_candidates, local_cable_prediction,
                                               cable_crop_boxes)
from easyhec.segmentation.mask_editor import MaskEditor
import test_mask_editor


class CompositionTests(unittest.TestCase):
    def test_strict_post_and_rail_exclusions_survive_silhouette_repair(self):
        robot = np.zeros((120, 160), bool)
        robot[10:110, 20:140] = True
        post = np.zeros_like(robot)
        post[10:110, 75:85] = True
        rail = np.zeros_like(robot)
        rail[70:73, 20:140] = True
        result, stats = compose_mask(robot, exclusions=[rail],
                                     strict_exclusions=[post, rail], hands=[robot])
        self.assertFalse(result[post | rail].any())
        self.assertTrue(result[30, 40])
        self.assertGreater(stats['strict_removed_px'], 0)

    def test_scene_background_opens_gap_without_removing_surrounding_robot(self):
        robot = np.zeros((120,160), bool)
        robot[10:110,20:140] = True
        wall = np.zeros_like(robot)
        wall[40:75,65:90] = True
        result, metadata = compose_mask(robot, backgrounds=[wall], repair_body=False)
        self.assertFalse(result[55,75])
        self.assertTrue(result[55,40])
        self.assertGreater(metadata['background_removed_px'], 0)
        # A background detection covering the whole robot is not trustworthy.
        result, metadata = compose_mask(robot, backgrounds=[np.ones_like(robot)])
        np.testing.assert_array_equal(result, robot)
        self.assertEqual(metadata['background_removed_px'], 0)

    def setUp(self):
        self.robot = np.zeros((120,160),bool)
        self.robot[10:40,30:130] = True
        self.robot[40:100,30:50] = True
        self.robot[40:100,110:130] = True
        self.post = np.zeros_like(self.robot)
        self.post[35:95,75:85] = True
        self.cable = np.zeros_like(self.robot)
        self.cable[50:56,45:70] = True
        self.raw = self.robot | self.cable
        self.cleanup = dict(prompts=dict(include=['vertical metal support'], exclude=['cable']),
                            additions=[self.post], exclusions=[self.cable])

    def test_adds_missing_support_and_removes_cable(self):
        result, stats = compose_mask(self.raw,[self.post],[self.cable])
        self.assertTrue(result[80,80])
        self.assertFalse(result[52,60])
        self.assertTrue(result[25,80])
        self.assertGreater(stats['added_px'],0)
        self.assertGreater(stats['removed_px'],0)

    def test_disconnected_and_oversized_additions_rejected(self):
        remote = np.zeros_like(self.raw)
        remote[100:110,140:150] = True
        result, stats = compose_mask(self.raw,[remote,np.ones_like(remote)])
        np.testing.assert_array_equal(result,self.raw)
        self.assertEqual(stats['skipped_additions'],2)

    def test_excessive_removal_and_empty_robot_unchanged(self):
        result, stats = compose_mask(self.raw,exclusions=[self.raw])
        np.testing.assert_array_equal(result,self.raw)
        self.assertEqual(stats['skipped_exclusions'],1)
        result,_ = compose_mask(np.zeros_like(self.raw),[self.post],[self.cable])
        self.assertFalse(result.any())

    def test_no_detections_and_wrong_shapes(self):
        result,_ = compose_mask(self.raw)
        np.testing.assert_array_equal(result,self.raw)
        with self.assertRaises(ValueError):
            compose_mask(self.raw,[self.post[:5]])
        with self.assertRaises(ValueError):
            compose_mask(self.raw,exclusions=[self.cable[:5]])

    def test_recovers_missing_hand_at_wrist_without_filling_gaps_or_screen(self):
        robot = np.zeros((400, 1280), bool)
        robot[40:160, 100:180] = True
        hand = np.zeros_like(robot)
        hand[160:280, 100:200] = True
        hand[210:240, 130:155] = False
        screen = np.zeros_like(robot)
        screen[160:170, 600:610] = True
        distant = np.roll(hand, 700, axis=1)
        result, stats = compose_mask(robot, hands=[hand, screen, distant], recover_hands=True)
        self.assertTrue(result[180, 150])
        self.assertFalse(result[220, 140])
        self.assertFalse(result[165, 605])
        self.assertFalse(result[180, 850])
        self.assertGreater(stats['hand_added_px'], 0)
        result, _ = compose_mask(robot, hands=[hand], recover_hands=False)
        self.assertFalse(result[180, 150])
        # Neither prompt sees a short wrist adapter; retain the hand but do
        # not fabricate foreground across that gap.
        separated = np.roll(hand, 40, axis=0)
        result, _ = compose_mask(robot, hands=[separated], recover_hands=True)
        self.assertTrue(result[230, 150])
        self.assertFalse(result[180, 150])

    def test_repairs_body_hole_but_protects_same_size_curled_hand_gap(self):
        robot = np.zeros((240, 1280), bool)
        robot[40:200, 60:200] = True
        robot[75:100, 100:125] = False
        robot[40:200, 300:440] = True
        robot[75:100, 340:365] = False
        hand = np.zeros_like(robot)
        hand[:, 300:440] = robot[:, 300:440]
        result, stats = compose_mask(robot, hands=[hand])
        self.assertTrue(result[85, 110])
        self.assertFalse(result[85, 350])
        self.assertEqual(stats['protected_hands'], 1)
        self.assertGreater(stats['silhouette_repair']['body_repair_px'], 0)
        # Even an overlapping support prediction must not fill the hand gap.
        support = hand.copy()
        support[75:100, 340:365] = True
        result, _ = compose_mask(robot, additions=[support], hands=[hand])
        self.assertFalse(result[85, 350])
        # A hand on a monitor cannot enable repair on the selected robot.
        remote = np.roll(hand, 500, axis=1)
        result, stats = compose_mask(robot, hands=[remote])
        self.assertFalse(result[85, 110])
        self.assertEqual(stats['protected_hands'], 0)

    def test_manual_restore_and_erase_survive_regeneration_and_toggle_undo(self):
        editor = MaskEditor(self.raw,cleanup=self.cleanup)
        self.assertTrue(editor.result()[80,80])
        self.assertFalse(editor.result()[52,60])
        editor.begin_stroke(60,52,1,2)
        editor.begin_stroke(80,80,0,2)
        editor.regenerate(self.raw)
        self.assertTrue(editor.result()[52,60])
        self.assertFalse(editor.result()[80,80])
        editor.checkpoint()
        editor.auto_enabled = False
        self.assertFalse(editor.result()[70,80])
        editor.undo()
        self.assertTrue(editor.result()[70,80])
        self.assertTrue(editor.metadata()['image_cleanup']['enabled'])


class SilhouetteRepairTests(unittest.TestCase):
    def test_loop_prompts_target_thin_enclosures_not_fingers_or_body_holes(self):
        mask = np.zeros((400, 1280), np.uint8)
        mask[40:160, 100:300] = 1
        mask[75:100, 160:190] = 0  # A hole through a thick body.
        mask[150:220, 200:208] = 1
        mask[150:220, 130:138] = 1
        mask[212:220, 130:208] = 1  # A thin U-shaped cable.
        mask[150:280, 280:295] = 1  # An isolated narrow finger.
        candidates = cable_loop_candidates(mask)
        self.assertTrue(candidates)
        for (x, y), region in candidates:
            self.assertTrue(mask[y, x])
            self.assertTrue(region[y, x])
            self.assertTrue(130 <= x < 208 and 160 <= y < 220)

    def test_local_prediction_rejects_body_and_distant_objects(self):
        robot = np.ones((100, 100), bool)
        local = np.zeros_like(robot)
        local[40:60, 40:60] = True
        cable = np.zeros_like(robot)
        cable[48:52, 42:58] = True
        self.assertTrue(local_cable_prediction(cable, (50, 50), local, robot))
        self.assertFalse(local_cable_prediction(robot, (50, 50), local, robot))
        cable[1:10, 1:10] = True
        self.assertFalse(local_cable_prediction(cable, (50, 50), local, robot))

    def test_local_prediction_rejects_thick_finger_even_within_area_and_location_limits(self):
        robot = np.ones((400, 1280), bool)
        finger = np.zeros_like(robot)
        finger[100:200, 100:140] = True
        self.assertFalse(local_cable_prediction(finger, (120, 150), robot, robot))

    def test_cable_crossing_solid_body_preserved_but_exposed_tail_removed(self):
        mask = np.zeros((240, 1280), bool)
        mask[40:200, 60:200] = True
        cable = np.zeros_like(mask)
        cable[110:116, 80:270] = True
        result, stats = repair_silhouette(mask | cable, cable)
        self.assertTrue(result[113, 130])
        self.assertFalse(result[113, 245])
        self.assertGreater(stats['cable_crossing_px'], 0)

    def test_cable_parallel_to_body_edge_is_not_restored_by_closing(self):
        robot = np.zeros((240, 1280), bool)
        robot[40:200, 60:200] = True
        cable = np.zeros_like(robot)
        cable[70:170, 203:207] = True
        cable[70:74, 195:207] = True
        cable[166:170, 195:207] = True
        result, _ = repair_silhouette(robot | cable, cable)
        self.assertFalse(result[100, 205])
        self.assertTrue(result[100, 190])

    def test_cable_crops_target_robot_neighborhood_and_stay_in_image(self):
        robot = np.zeros((800, 1280), bool)
        robot[200:400, 100:300] = True
        cable = np.zeros_like(robot)
        cable[250:350, 290:295] = True
        remote = np.zeros_like(robot)
        remote[600:700, 1100:1105] = True
        boxes = cable_crop_boxes(robot, [cable, remote])
        self.assertEqual(len(boxes), 1)
        x0, y0, x1, y1 = boxes[0]
        self.assertTrue(0 <= x0 < 290 < x1 <= 1280)
        self.assertTrue(0 <= y0 < 250 < 350 < y1 <= 800)
        self.assertEqual(cable_crop_boxes(robot, []), [])

    def test_enclosed_and_open_finger_gaps_remain_background(self):
        mask = np.zeros((240, 1280), bool)
        mask[40:200, 60:200] = True
        mask[75:100, 100:125] = False
        mask[150:180, 420:460] = True
        for x in [420, 436, 452]:
            mask[50:150, x:x+8] = True
        result, stats = repair_silhouette(mask, np.zeros_like(mask))
        self.assertFalse(result[85, 110])
        self.assertFalse(result[100, 432])
        self.assertFalse(result[100, 448])
        self.assertFalse(result[20, 110])
        self.assertEqual(stats['body_repair_px'], 0)

    def test_support_cleanup_preserves_open_gaps_between_thick_fingers(self):
        robot = np.zeros((240, 1280), bool)
        robot[40:200, 60:200] = True
        robot[40:145, 100:108] = False
        robot[40:145, 150:158] = False
        support = np.zeros_like(robot)
        support[190:220, 120:140] = True
        result, _ = compose_mask(robot, additions=[support])
        self.assertTrue(result[210, 130])
        self.assertFalse(result[140, 104])
        self.assertFalse(result[140, 154])
        self.assertFalse((result & ~robot & ~support).any())

    def test_cable_loop_background_not_filled(self):
        mask = np.zeros((240, 1280), bool)
        mask[30:90, 60:200] = True
        cable = np.zeros_like(mask)
        cable[85:180, 80:86] = True
        cable[85:180, 170:176] = True
        cable[174:180, 80:176] = True
        result, _ = repair_silhouette(mask | cable, cable)
        self.assertFalse(result[140, 120])
        self.assertFalse(result[150, 83])

    def test_manual_body_hole_remains_erased_after_regeneration(self):
        raw = np.zeros((240, 1280), np.uint8)
        raw[40:200, 60:200] = 1
        raw[75:100, 100:125] = 0
        editor = MaskEditor(raw, cleanup=dict(prompts={}, additions=[], exclusions=[]))
        self.assertFalse(editor.result()[85, 110])
        editor.begin_stroke(110, 85, 0, 4)
        editor.regenerate(raw)
        self.assertFalse(editor.result()[85, 110])

    def test_open_image_border_and_large_gaps_are_preserved(self):
        mask = np.ones((240, 1280), bool)
        mask[:80, 100:150] = False
        mask[100:220, 300:450] = False
        result, _ = repair_silhouette(mask, np.zeros_like(mask))
        self.assertFalse(result[40, 125])
        self.assertFalse(result[150, 370])


class ReviewTests(unittest.TestCase):
    run_review = test_mask_editor.ReviewTests.run_review

    def test_toggle_changes_and_undo_then_accept_with_cached_detections(self):
        fixture = CompositionTests()
        fixture.setUp()
        adapter = Mock(text='robot', object_id=1, return_value=fixture.raw.astype(np.uint8))
        adapter.cleanup_prompts = fixture.cleanup['prompts']
        adapter.detect_objects.side_effect = [[], [], [fixture.post]]
        adapter.detect_exclusions.return_value = [fixture.cable]
        result = self.run_review(adapter,[([],ord('d')),([],ord('a')),([],ord('u')),([],ord('t'))])
        self.assertTrue(result[80,80])
        self.assertFalse(result[52,60])
        self.assertEqual(adapter.detect_objects.call_count,3)
        self.assertEqual(adapter.detect_objects.call_args_list[0].args, (['robot hand', 'black glove'],))
        self.assertEqual(adapter.detect_exclusions.call_count,1)
        self.assertTrue(adapter.review_metadata['image_cleanup']['enabled'])


if __name__ == '__main__':
    unittest.main()
