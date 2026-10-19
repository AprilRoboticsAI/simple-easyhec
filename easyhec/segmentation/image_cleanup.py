"""Image-only composition of robot, support, and unwanted-object detections."""
import cv2
import numpy as np


def cable_loop_candidates(mask):
    """Propose point prompts on thin enclosures, without removing any pixels.

    A small bounded background gap must touch a foreground fragment lost by
    opening. Ordinary fingertips do not enclose background, and holes through
    thick parts do not touch these fragments. SAM must still segment and pass
    the locality checks before a candidate becomes an exclusion.
    """
    mask = np.asarray(mask, bool)
    scale = mask.shape[1] / 1280
    radius = max(1, round(12 * scale))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*radius+1,)*2)
    _, holes, stats, _ = cv2.connectedComponentsWithStats((~mask).astype(np.uint8), 8)
    valid = (stats[:, 4] >= 128*scale**2) & (stats[:, 4] <= 5000*scale**2)
    valid[np.unique(np.r_[0, holes[0], holes[-1], holes[:, 0], holes[:, -1]])] = False
    near_hole = cv2.dilate(valid[holes].astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    opened = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel) > 0
    distance = cv2.distanceTransform((~opened).astype(np.uint8), cv2.DIST_L2, 5)
    interior = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    count, labels, stats, _ = cv2.connectedComponentsWithStats((mask & ~opened).astype(np.uint8), 8)
    candidates, alternatives = [], []
    for index in range(1, count):
        if not 40*scale**2 < stats[index, 4] < 5000*scale**2:
            continue
        component = labels == index
        if np.count_nonzero(component & near_hole) <= max(2, 8*scale):
            continue
        # Prefer the interior of the cable, away from both its image boundary
        # and the thick robot body. Edge clicks can select several nearby wires.
        y, x = np.unravel_index(np.argmax(distance * interior * component), mask.shape)
        # A shallow edge nick is not enough evidence for an automatic click.
        if distance[y, x] < max(2, 5*scale):
            continue
        # Include the enclosed gap so a thick connector on the opposite side
        # of a loop is still local to the prompt's thin side.
        touching = np.unique(holes[cv2.dilate(component.astype(np.uint8),
                                            np.ones((3, 3), np.uint8)) > 0])
        touching = touching[valid[touching]]
        enclosure = component | np.isin(holes, touching)
        region = cv2.dilate(enclosure.astype(np.uint8),
                           cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4*radius+1,)*2)) > 0
        candidates.append(((int(x), int(y)), region))
        # A connector and its wire can be separate SAM objects. Offer a second
        # interior point on a distant section; skip it if the first mask covers it.
        yy, xx = np.ogrid[:mask.shape[0], :mask.shape[1]]
        distant = component & ((xx-x)**2 + (yy-y)**2 > (40*scale)**2)
        if distant.any():
            y2, x2 = np.unravel_index(np.argmax(distance * interior * distant), mask.shape)
            if distance[y2, x2] >= max(2, 5*scale):
                alternatives.append(((int(x2), int(y2)), region))
    return candidates[:6] + alternatives[:6]


def local_cable_prediction(mask, point, region, robot):
    """Reject point predictions selecting a whole hand, arm, or distant object."""
    mask = np.asarray(mask, bool)
    x, y = point
    area = np.count_nonzero(mask)
    return (mask[y, x] and 0 < area <= .03 * np.count_nonzero(robot)
            and np.count_nonzero(mask & region) >= .5 * area
            and cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5).max()
            <= max(2, 14 * mask.shape[1] / 1280))


def repair_silhouette(foreground, excluded):
    """Keep cable crossings inside solid parts and repair supported body cavities.

    Closing is only a proposal: restoring a cable cut is restricted to the
    pre-subtraction mask. New foreground requires a small connected cavity with
    at least 72% foreground support in a surrounding band. This prevents global
    hole filling from turning cable loops or narrow finger gaps into foreground.
    Parameters scale from 1280-pixel images; no pose or rendering is involved.
    """
    foreground, excluded = np.asarray(foreground, bool), np.asarray(excluded, bool)
    if foreground.ndim != 2 or excluded.shape != foreground.shape:
        raise ValueError('Foreground and exclusion dimensions must match')
    scale = foreground.shape[1] / 1280
    radius, band = max(1, round(10 * scale)), max(1, round(12 * scale))
    max_area = max(1, round(8000 * scale**2))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*radius+1,)*2)
    cut = foreground & ~excluded
    # Pad explicitly so the image border cannot close open background regions.
    padded = np.pad(cut.astype(np.uint8), radius + 1)
    closed = cv2.morphologyEx(padded, cv2.MORPH_CLOSE, kernel)
    closed = closed[radius+1:-(radius+1), radius+1:-(radius+1)] > 0
    restored = closed & foreground & excluded
    result = cut | restored
    _, labels, stats, _ = cv2.connectedComponentsWithStats((~closed).astype(np.uint8), 8)
    fill = stats[:, cv2.CC_STAT_AREA] <= max_area
    fill[np.unique(np.r_[0, labels[0], labels[-1], labels[:, 0], labels[:, -1]])] = False
    candidates = (closed | fill[labels]) & ~result
    # Do not create new foreground in the background enclosed by cable loops.
    # Cable crossings already present in the mask were restored above.
    cable_neighborhood = cv2.dilate(excluded.astype(np.uint8), kernel) > 0
    candidates &= ~cable_neighborhood
    count, labels, stats, _ = cv2.connectedComponentsWithStats(candidates.astype(np.uint8), 8)
    ring_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*band+1,)*2)
    repaired = np.zeros_like(result)
    for index in range(1, count):
        x, y, width, height, area = stats[index]
        if area > max_area:
            continue
        # Exclude clipped components: their surrounding support is unknown.
        if x < band or y < band or x+width+band > result.shape[1] or y+height+band > result.shape[0]:
            continue
        region = np.s_[y-band:y+height+band, x-band:x+width+band]
        cavity = labels[region] == index
        ring = (cv2.dilate(cavity.astype(np.uint8), ring_kernel) > 0) & ~cavity
        support = np.count_nonzero(ring & cut[region]) / max(1, ring.sum())
        if support >= .72:
            repaired[region] |= cavity
    if repaired.sum() > .1 * max(1, foreground.sum()):
        repaired.fill(False)
    result |= repaired
    return result, dict(cable_crossing_px=int((result & foreground & excluded).sum()),
                        body_repair_px=int((result & ~foreground).sum()),
                        closing_radius_px=radius, support_band_px=band,
                        max_cavity_area_px=max_area, min_boundary_support=.72)


def compose_mask(robot, additions=(), exclusions=()):
    """Add nearby parts, remove exposed cables, and repair body cavities.

    Reject disconnected additions (e.g. objects on a monitor) and gross changes.
    These are review safeguards, not guarantees of segmentation correctness.
    """
    robot = np.asarray(robot, bool)
    if robot.ndim != 2:
        raise ValueError('Expected a 2D robot mask')
    result = robot.copy()
    added, removed = np.zeros_like(robot), np.zeros_like(robot)
    stats = dict(accepted_additions=0, skipped_additions=0, accepted_exclusions=0,
                 skipped_exclusions=0)
    radius = max(1, round(12 * robot.shape[1] / 1280))
    near = cv2.dilate(robot.astype(np.uint8),
                      cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*radius+1,)*2)) > 0
    area = robot.sum()
    for obj in additions:
        obj = np.asarray(obj, bool)
        if obj.shape != robot.shape:
            raise ValueError('Object and robot dimensions must match')
        novel = obj & ~robot
        if not area or not (obj & near).any() or (added | novel).sum() > .25 * area:
            stats['skipped_additions'] += 1
            continue
        added |= novel
        stats['accepted_additions'] += 1
    result |= added
    for obj in exclusions:
        obj = np.asarray(obj, bool)
        if obj.shape != robot.shape:
            raise ValueError('Object and robot dimensions must match')
        proposal = obj & result
        if not area or (removed | proposal).sum() > .25 * area:
            stats['skipped_exclusions'] += 1
            continue
        removed |= proposal
        stats['accepted_exclusions'] += 1
    result, repair = repair_silhouette(result, removed)
    return result, dict(stats, silhouette_repair=repair, added_px=int((result & ~robot).sum()),
                        removed_px=int((robot & ~result).sum()))
