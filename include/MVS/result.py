"""Common return type for depth backends."""
import collections

# depth, confidence : HxW float32 (metres, 0..1)
# ref_index         : which view of the window the depth belongs to
# K                 : 3x3 intrinsics matching the depth map's resolution.
#                     Backends may resize, so this is NOT necessarily the K that
#                     arrived in the message.
# backend           : provenance string, carried into DepthMsg for benchmarking
DepthResult = collections.namedtuple(
    "DepthResult", ["depth", "confidence", "ref_index", "K", "backend"])
