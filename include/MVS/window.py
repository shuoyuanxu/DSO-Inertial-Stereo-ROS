"""Assembly of DSO sliding windows from SlidingWindowsMsg.

The reference implementation kept a single in-flight window in a global list and
cleared it whenever a message with a new msg_id arrived. That drops a window
entirely if messages interleave, and it silently mixes views when a reset
renumbers ids. It also always used view 0 as the MVS reference, which is the
*oldest* keyframe in DSO's window - the one with the least overlap with the rest.

This keeps a small dict of partial windows keyed by msg_id, completes a window
only when all window_size views are present, and picks the reference view
explicitly.
"""
import numpy as np


class Window(object):
    """A completed sliding window: views sorted by index."""

    def __init__(self, msg_id, views):
        self.msg_id = msg_id
        self.views = views                     # list of SlidingWindowsMsg, index-sorted
        self.depth_min = float(views[0].depth_min)
        self.depth_max = float(views[0].depth_max)

    def __len__(self):
        return len(self.views)

    def pose(self, i):
        """Rigid cam->world 4x4 of view i."""
        return np.array(self.views[i].camToWorld, dtype=np.float64).reshape(4, 4)

    def K(self, i):
        """3x3 intrinsics of view i, in pixels of the published image."""
        fx, fy, cx, cy = self.views[i].Intrinsics
        K = np.eye(3, dtype=np.float64)
        K[0, 0] = fx
        K[1, 1] = fy
        K[0, 2] = cx
        K[1, 2] = cy
        return K

    def centres(self):
        return np.array([self.pose(i)[:3, 3] for i in range(len(self.views))])

    def baseline_span(self):
        """Largest distance between any two camera centres, in metres.

        Windows during DSO init have all views at nearly the same place; a plane
        sweep over those produces confident nonsense, so callers gate on this.
        """
        c = self.centres()
        if len(c) < 2:
            return 0.0
        return float(np.linalg.norm(c.max(0) - c.min(0)))

    def ref_index(self):
        """Index of the view to reconstruct.

        The middle keyframe, not view 0: it has the most overlap with the rest of
        the window on both sides, which is what a plane sweep wants. Using the
        oldest frame (the reference implementation's choice) means half the
        window is already out of its field of view on a forward-moving camera.
        """
        return len(self.views) // 2


class WindowAssembler(object):
    def __init__(self, max_pending=8):
        self._pending = {}          # msg_id -> {index: msg}
        self._max_pending = max_pending
        self.dropped = 0            # windows evicted incomplete, for diagnostics

    def add(self, msg):
        """Feed one message. Returns a Window once that window is complete, else None."""
        if msg.window_size == 0:
            return None
        slot = self._pending.setdefault(msg.msg_id, {})
        slot[msg.index] = msg

        if len(slot) < msg.window_size:
            self._evict()
            return None

        del self._pending[msg.msg_id]
        views = [slot[i] for i in sorted(slot)]
        return Window(msg.msg_id, views)

    def _evict(self):
        """Bound memory: msg_id increases monotonically, so drop the oldest."""
        while len(self._pending) > self._max_pending:
            oldest = min(self._pending)
            del self._pending[oldest]
            self.dropped += 1
