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
    def __init__(self, mask, object_id=None, pending=False):
        self.raw = (mask > 0).copy()
        self.object_id = object_id
        self.points = []
        self.pending = pending
        self.overrides = np.full(mask.shape, -1, np.int8)
        self.strokes = []
        self.history = []
        self._cache = None

    def checkpoint(self):
        self.history.append((self.raw.copy(), self.overrides.copy(), self.object_id,
                             deepcopy(self.points), self.pending,
                             deepcopy(self.strokes)))
        del self.history[:-32]
        self._cache = None

    def undo(self):
        if self.history:
            (self.raw, self.overrides, self.object_id, self.points, self.pending,
             self.strokes) = self.history.pop()
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
            result = fill_small_holes(self.raw)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            result = cv2.morphologyEx(result.astype(np.uint8), cv2.MORPH_OPEN, kernel) > 0
            seeds = [(x, y) for x, y, label in self.points if label > 0]
            result = remove_small_islands(result, foreground_points=seeds)
            manual = self.overrides >= 0
            result[manual] = self.overrides[manual] > 0
            self._cache = result
        return self._cache.copy()

    def metadata(self):
        return dict(version=1, cleanup=dict(islands=True, holes=True, opening=True, min_island_area_px=32,
                    max_hole_area_px=128, opening_radius_px=2),
                    cleanup_order=['holes', 'opening', 'islands'],
                    brush_applied_after_cleanup=True, strokes=deepcopy(self.strokes))


def review_mask(image, segmenter):
    """Review either text- or point-initialized SAM masks in a single window."""
    text = segmenter.text
    initial = segmenter(image, []) if text else np.zeros(image.shape[:2], np.uint8)
    editor = MaskEditor(initial, segmenter.object_id if text else 1, pending=not bool(text))
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
          '[/]: brush size; cleanup is automatic; '
          'u: undo; v: raw SAM comparison; r: reset all edits; Esc: cancel.')
    try:
        while True:
            display = image.copy()
            selected = editor.raw if original else editor.result()
            display[selected] = (.55 * display[selected] + .45 * np.array([40, 230, 40])).astype(np.uint8)
            if not original:
                for x, y, label in editor.points:
                    cv2.circle(display, (x, y), 4, (40, 230, 40) if label > 0 else (230, 40, 40), -1)
                if brush_mode and cursor is not None:
                    cv2.circle(display, cursor, radius, (255, 255, 255), 1)
            status = ('RAW SAM - v: edited preview' if original else
                      't: regenerate SAM (pending clicks)' if editor.pending else 't: accept mask')
            mode = f'BRUSH: left erase / right restore, radius {radius}px [ / ]' if brush_mode else 'SAM: left positive / right negative'
            cleanup = 'Auto cleanup: small islands, small holes, gentle opening'
            for row, line in enumerate([status, mode, cleanup, 'b: brush | e: clicks | u: undo | v: compare | r: reset | Esc: cancel']):
                y = 23 + row * 24
                cv2.putText(display, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 0, 0), 3)
                cv2.putText(display, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 1)
            cv2.imshow(window, cv2.cvtColor(display, cv2.COLOR_RGB2BGR))
            key = cv2.waitKey(20) & 0xff
            if key == 27 or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                raise ValueError('Annotation cancelled; current mask was not saved')
            if key == 255:
                continue
            dragging = False
            if key == ord('v'):
                original = not original
                continue
            if original:
                # Return to the edited preview before any action, especially accept.
                original = False
                continue
            if key == ord('t'):
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
