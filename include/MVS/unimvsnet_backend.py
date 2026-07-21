"""UniMVSNet depth backend.

Wraps the upstream network (python/dense_depth/networks/, copied unmodified)
in a minimal inference path. The upstream Model class is deliberately not used:
it drags in tensorboardX, progressbar, dataset loaders, the loss and distributed
init, none of which an inference node needs.

Cascade conventions that matter and are easy to get wrong:
  * MVSNet treats view 0 as the reference view, so the window is reordered to
    put the chosen reference first.
  * proj_matrices are given at STAGE-1 resolution (1/4), because
    stage_scale = 2**(3-stage_idx-1) is 4/2/1 for stages 1/2/3. The node scales
    them back up by 2 and 4 for stages 2 and 3. The intrinsics/4 in the
    reference implementation is this pyramid factor, NOT a fix for a resolution
    mismatch - dropping it silently destroys the sweep geometry.
  * numdepth=192 with ndepths=[48,32,8] and interval_ratio=[4,2,1] is a matched
    set: stage 1 covers the full range at 4x the base interval.
"""
import os
import numpy as np
import torch

from .networks.mvsnet import MVSNet
from .result import DepthResult

# stage-1 is at 1/4 of the input resolution
STAGE1_DOWNSAMPLE = 4.0
# the network's feature pyramid requires both image dims to be multiples of this
SIZE_BASE = 32


class UniMVSNetBackend(object):
    name = "unimvsnet"

    def __init__(self, ckpt, ndepths=(48, 32, 8), interval_ratio=(4, 2, 1),
                 numdepth=192, max_w=640, max_h=480, device=None,
                 fea_mode="fpn", agg_mode="variance", depth_mode="unification",
                 fp16=False):
        self.numdepth = int(numdepth)
        self.max_w = int(max_w)
        self.max_h = int(max_h)
        self.fp16 = bool(fp16)
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu"))

        self.net = MVSNet(ndepths=list(ndepths),
                          depth_interval_ratio=list(interval_ratio),
                          fea_mode=fea_mode, agg_mode=agg_mode,
                          depth_mode=depth_mode).to(self.device)

        if not os.path.isfile(ckpt):
            raise IOError("UniMVSNet checkpoint not found: %s" % ckpt)
        state = torch.load(ckpt, map_location="cpu")
        weights = state["model"] if "model" in state else state
        missing, unexpected = self.net.load_state_dict(weights, strict=False)
        # strict=False is upstream's behaviour, but silently loading nothing is a
        # real failure mode - surface how much actually matched
        loaded = len(self.net.state_dict()) - len(missing)
        if loaded == 0:
            raise RuntimeError("checkpoint %s matched no parameters" % ckpt)
        self.load_report = "loaded %d/%d tensors (%d missing, %d unexpected)" % (
            loaded, len(self.net.state_dict()), len(missing), len(unexpected))

        self.net.eval()
        if self.fp16:
            self.net.half()

    def _fit(self, img, K):
        """Resize to something the pyramid accepts, adjusting K to match."""
        h, w = img.shape[:2]
        scale = min(1.0, float(self.max_h) / h, float(self.max_w) / w)
        new_w = int(scale * w) // SIZE_BASE * SIZE_BASE
        new_h = int(scale * h) // SIZE_BASE * SIZE_BASE
        if new_w <= 0 or new_h <= 0:
            raise ValueError("image %dx%d too small for base %d" % (w, h, SIZE_BASE))
        if (new_w, new_h) != (w, h):
            import cv2
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            K = K.copy()
            K[0, :] *= float(new_w) / w
            K[1, :] *= float(new_h) / h
        return img, K

    @torch.no_grad()
    def run(self, window, images):
        """window: Window. images: list of HxWx3 float32 in [0,1], index-aligned.

        Returns DepthResult. K_used is the reference intrinsics AFTER any resize,
        so it always matches the returned depth map - publishing the original K
        with a resized depth map silently scales the whole reconstruction.
        """
        ref = window.ref_index()
        # MVSNet uses view 0 as the reference: put our chosen reference first,
        # keeping the others in their original temporal order.
        order = [ref] + [i for i in range(len(window)) if i != ref]

        imgs, projs = [], []
        K_ref = None
        for i in order:
            img, K = self._fit(images[i], window.K(i))
            if K_ref is None:
                K_ref = K
            # world->cam; the message carries cam->world
            extrinsic = np.linalg.inv(window.pose(i))
            K_stage1 = K.copy()
            K_stage1[:2, :] /= STAGE1_DOWNSAMPLE

            p = np.zeros((2, 4, 4), dtype=np.float32)
            p[0] = extrinsic
            p[1, :3, :3] = K_stage1
            imgs.append(img)
            projs.append(p)

        imgs = np.stack(imgs).transpose([0, 3, 1, 2])[None]      # (1,V,3,H,W)
        projs = np.stack(projs)                                   # (V,2,4,4)

        stage2 = projs.copy(); stage2[:, 1, :2, :] *= 2
        stage3 = projs.copy(); stage3[:, 1, :2, :] *= 4
        proj_ms = {
            "stage1": torch.from_numpy(projs[None]).to(self.device),
            "stage2": torch.from_numpy(stage2[None]).to(self.device),
            "stage3": torch.from_numpy(stage3[None]).to(self.device),
        }

        # linear depth hypotheses over the range DSO measured for this window
        dv = np.linspace(window.depth_min, window.depth_max,
                         self.numdepth, dtype=np.float32)[None]

        t_imgs = torch.from_numpy(imgs).to(self.device)
        t_dv = torch.from_numpy(dv).to(self.device)
        if self.fp16:
            t_imgs = t_imgs.half()

        out = self.net(t_imgs, proj_ms, t_dv)
        depth = out["depth"][0].float().cpu().numpy()
        conf = out["photometric_confidence"][0].float().cpu().numpy()
        del out
        torch.cuda.empty_cache()

        return DepthResult(depth=depth.astype(np.float32),
                           confidence=conf.astype(np.float32),
                           ref_index=ref, K=K_ref, backend=self.name)
