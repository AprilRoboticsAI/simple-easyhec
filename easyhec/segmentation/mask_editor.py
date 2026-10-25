"""Deterministic mask cleanup and brush edits, independent of the SAM predictor."""
from copy import deepcopy

import cv2
import numpy as np


def remove_small_islands(mask, min_area=32, foreground_points=()):
    """Remove small selected components, preserving explicitly selected islands."""
    _, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    keep = stats[:, cv2.CC_STAT_AREA] >= min_area
    for x, y in foreground_points:
        keep[labels[y, x]] = True
    keep[0] = False
    return keep[labels]


def fill_small_holes(mask, max_area=128):
    """Fill only enclosed small holes; image-border background is never filled."""
    mask = np.asarray(mask, dtype=bool)
    _, labels, stats, _ = cv2.connectedComponentsWithStats((~mask).astype(np.uint8), connectivity=8)
    fill = stats[:, cv2.CC_STAT_AREA] <= max_area
    fill[np.unique(np.r_[0, labels[0], labels[-1], labels[:, 0], labels[:, -1]])] = False
    return mask | fill[labels]


class MaskEditor:
    """Cleanup is recomputed from raw SAM; manual strokes always apply last.

    Undo snapshots are bounded to 32 user actions. Stroke coordinates are saved
    for provenance/replay, with radii in original image pixels.
    """
    def __init__(self, mask, object_id=None, pending=False, cleanup=None):
        self.raw = (mask > 0).copy()
        self.object_id = object_id
        self.points = []
        self.pending = pending
        self.overrides = np.full(mask.shape, -1, np.int8)
        self.strokes = []
        self.history = []
        self._cache = None
        self.cleanup = cleanup
        self.auto_enabled = cleanup is not None
        self.auto_metadata = None
        self.auto_added = np.zeros_like(self.raw)
        self.auto_removed = np.zeros_like(self.raw)

    def checkpoint(self):
        self.history.append((self.raw.copy(), self.overrides.copy(), self.object_id,
                             deepcopy(self.points), self.pending,
                             deepcopy(self.strokes), self.auto_enabled))
        del self.history[:-32]
        self._cache = None

    def undo(self):
        if self.history:
            (self.raw, self.overrides, self.object_id, self.points, self.pending,
             self.strokes, self.auto_enabled) = self.history.pop()
            self._cache = None

    def add_point(self, x, y, label):
        self.checkpoint()
        self.points.append((x, y, label))
        self.pending = True

    def regenerate(self, mask, object_id=None):
        self.checkpoint()
        self.raw = (mask > 0).copy()
        self.object_id = object_id
        self.pending = False

    def reset(self, mask, object_id=None, pending=False):
        self.checkpoint()
        self.raw = (mask > 0).copy()
        self.object_id = object_id
        self.points = []
        self.pending = pending
        self.overrides.fill(-1)
        self.strokes = []

    def begin_stroke(self, x, y, value, radius):
        self.checkpoint()
        self.strokes.append(dict(value=value, radius=radius, xy=[[x, y]]))
        # OpenCV drawing does not support signed 8-bit arrays.
        brush = np.zeros(self.raw.shape, np.uint8)
        cv2.circle(brush, (x, y), radius, 1, -1)
        self.overrides[brush > 0] = value

    def extend_stroke(self, x, y):
        stroke = self.strokes[-1]
        prev = tuple(stroke['xy'][-1])
        brush = np.zeros(self.raw.shape, np.uint8)
        cv2.line(brush, prev, (x, y), 1, 2 * stroke['radius'] + 1)
        cv2.circle(brush, (x, y), stroke['radius'], 1, -1)
        self.overrides[brush > 0] = stroke['value']
        stroke['xy'].append([x, y])
        self._cache = None

    def result(self):
        if self._cache is None:
            result = self.raw.copy()
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            result = cv2.morphologyEx(result.astype(np.uint8), cv2.MORPH_OPEN, kernel) > 0
            seeds = [(x, y) for x, y, label in self.points if label > 0]
            result = remove_small_islands(result, foreground_points=seeds)
            if self.cleanup is not None:
                from easyhec.segmentation.image_cleanup import compose_mask

                composed, self.auto_metadata = compose_mask(
                    result, self.cleanup['additions'], self.cleanup['exclusions'],
                    hands=self.cleanup.get('hands', ()),
                    recover_hands=self.cleanup['prompts'].get('recover_hands', False),
                    backgrounds=self.cleanup.get('backgrounds', ()),
                    strict_exclusions=self.cleanup.get('strict_exclusions', ()),
                    repair_body=self.cleanup['prompts'].get('repair_body', False))
                # Cable subtraction can detach fragments after the first island
                # pass. Filter those before recording changes or applying brushes.
                min_area = max(1, round(256 * self.raw.shape[1]**2 / 1280**2))
                filtered = remove_small_islands(composed, min_area=min_area,
                                                foreground_points=seeds)
                self.auto_metadata['final_islands_removed_px'] = int((composed & ~filtered).sum())
                self.auto_metadata['final_min_island_area_px'] = min_area
                composed = filtered
                self.auto_metadata['added_px'] = int((composed & ~result).sum())
                self.auto_metadata['removed_px'] = int((result & ~composed).sum())
                self.auto_added = composed & ~result
                self.auto_removed = result & ~composed
                if self.auto_enabled:
                    result = composed
            manual = self.overrides >= 0
            result[manual] = self.overrides[manual] > 0
            self._cache = result
        return self._cache.copy()

    def metadata(self):
        self.result()
        automatic = None if self.cleanup is None else dict(
            self.auto_metadata, enabled=self.auto_enabled, prompts=self.cleanup['prompts'])
        return dict(version=2, image_cleanup=automatic, cleanup=dict(islands=True, holes=False, opening=True, min_island_area_px=32,
                    max_hole_area_px=0, opening_radius_px=2),
                    cleanup_order=['opening', 'islands'],
                    brush_applied_after_cleanup=True, strokes=deepcopy(self.strokes))


def prepare_mask_editor(image, segmenter):
    """Build the same automatic mask for interactive review and offline inspection."""
    text = segmenter.text
    initial = segmenter(image, []) if text else np.zeros(image.shape[:2], np.uint8)
    prompts = getattr(segmenter, 'cleanup_prompts', None)
    cleanup = None
    if isinstance(prompts, dict):
        prompts = dict(prompts)
        prompts.setdefault('protect', ['robot hand', 'black glove'])
        prompts.setdefault('background', ['wall', 'table'])
        cleanup = dict(prompts=prompts,
                       hands=segmenter.detect_objects(prompts['protect']),
                       backgrounds=segmenter.detect_objects(prompts['background']),
                       additions=segmenter.detect_objects(prompts['include']),
                       strict_exclusions=(segmenter.detect_objects(prompts['strict_exclude'])
                                          if prompts.get('strict_exclude') else []),
                       exclusions=segmenter.detect_exclusions(prompts['exclude'], initial))
    return MaskEditor(initial, segmenter.object_id if text else 1, pending=not bool(text),
                      cleanup=cleanup)


def review_mask(image, segmenter):
    """Review either text- or point-initialized SAM masks in a single window."""
    text = segmenter.text
    editor = prepare_mask_editor(image, segmenter)
    show_changes = False
    window = 'SAM 3.1 mask review'
    brush_mode, original = False, False
    radius, cursor, dragging = 24, None, False

    def click(event, x, y, flags, param):
        nonlocal cursor, dragging
        cursor = (x, y)
        if event in (cv2.EVENT_LBUTTONUP, cv2.EVENT_RBUTTONUP):
            dragging = False
            return
        if original or not (0 <= x < image.shape[1] and 0 <= y < image.shape[0]):
            dragging = False
            return
        if event in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_RBUTTONDOWN):
            if brush_mode:
                # Erase on left drag, restore/add foreground on right drag.
                editor.begin_stroke(x, y, int(event == cv2.EVENT_RBUTTONDOWN), radius)
                dragging = True
            else:
                editor.add_point(x, y, 1 if event == cv2.EVENT_LBUTTONDOWN else -1)
        elif event == cv2.EVENT_MOUSEMOVE and dragging:
            if flags & (cv2.EVENT_FLAG_LBUTTON | cv2.EVENT_FLAG_RBUTTON):
                editor.extend_stroke(x, y)
            else:
                dragging = False

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, image.shape[1], image.shape[0])
    cv2.setMouseCallback(window, click)
    print('t: generate/accept; e: SAM clicks; b: brush (left erase, right restore); '
          '[/]: brush size; a: image cleanup on/off; d: show additions/removals; '
          'u: undo; v: raw SAM comparison; r: reset all edits; x: reject image; Esc: cancel.')
    try:
        while True:
            display = image.copy()
            selected = editor.raw if original else editor.result()
            display[selected] = (.55 * display[selected] + .45 * np.array([40, 230, 40])).astype(np.uint8)
            if show_changes and not original and editor.auto_enabled:
                added = editor.auto_added & selected
                removed = editor.auto_removed & ~selected
                display[added] = (.4 * display[added] + .6 * np.array([40, 220, 255])).astype(np.uint8)
                display[removed] = (.4 * display[removed] + .6 * np.array([230, 40, 230])).astype(np.uint8)
            if not original:
                for x, y, label in editor.points:
                    cv2.circle(display, (x, y), 4, (40, 230, 40) if label > 0 else (230, 40, 40), -1)
                if brush_mode and cursor is not None:
                    cv2.circle(display, cursor, radius, (255, 255, 255), 1)
            status = ('RAW SAM - v: edited preview' if original else
                      't: regenerate SAM (pending clicks)' if editor.pending else 't: accept mask')
            mode = f'BRUSH: left erase / right restore, radius {radius}px [ / ]' if brush_mode else 'SAM: left positive / right negative'
            cleanup_status = 'Auto cleanup: small islands, small holes, gentle opening'
            if editor.cleanup is not None:
                cleanup_status = ('a: image cleanup ' + ('ON' if editor.auto_enabled else 'OFF')
                                  + ' | d: changes (cyan added, magenta removed)')
            for row, line in enumerate([status, mode, cleanup_status, 'b: brush | e: clicks | u: undo | v: compare | r: reset | x: reject image | Esc: cancel']):
                y = 23 + row * 24
                cv2.putText(display, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 0, 0), 3)
                cv2.putText(display, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 1)
            cv2.imshow(window, cv2.cvtColor(display, cv2.COLOR_RGB2BGR))
            key = cv2.waitKey(20) & 0xff
            if key == 27 or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                raise ValueError('Annotation cancelled; current mask was not saved')
            if key == 255:
                continue
            if key == ord('x'):
                return None
            dragging = False
            if key == ord('v'):
                original = not original
                continue
            if original:
                # Return to the edited preview before any action, especially accept.
                original = False
                continue
            if key == ord('d') and editor.cleanup is not None:
                show_changes = not show_changes
            elif key == ord('a') and editor.cleanup is not None:
                editor.checkpoint()
                editor.auto_enabled = not editor.auto_enabled
            elif key == ord('t'):
                if editor.pending:
                    if not text and not editor.points:
                        print('Add a SAM point before generating a mask.')
                        continue
                    mask = segmenter(image, editor.points)
                    editor.regenerate(mask, segmenter.object_id if text else 1)
                elif editor.result().any():
                    segmenter.points = list(editor.points)
                    segmenter.object_id = editor.object_id
                    segmenter.review_metadata = editor.metadata()
                    return editor.result().astype(np.uint8)
                else:
                    print('Cannot accept an empty mask; restore pixels or change the prompts.')
            elif key == ord('b'):
                brush_mode = not brush_mode
            elif key == ord('e'):
                brush_mode = False
                editor.checkpoint()
                editor.pending = True
            elif key == ord('u'):
                editor.undo()
            elif key in (ord('['), ord(']')):
                radius = max(1, min(256, radius + (4 if key == ord(']') else -4)))
            elif key == ord('r'):
                mask = segmenter(image, []) if text else np.zeros(image.shape[:2], np.uint8)
                editor.reset(mask, segmenter.object_id if text else 1, pending=not bool(text))
    finally:
        cv2.destroyWindow(window)
