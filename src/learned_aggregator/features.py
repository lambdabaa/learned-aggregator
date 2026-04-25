import numpy as np


def extract_features(step_scores: list[float]) -> np.ndarray:
    """Convert variable-length step scores to a fixed 10-dim feature vector.

    Features (position indices normalised by trajectory length):
      0  mean
      1  min
      2  max
      3  last step score
      4  trajectory length (raw count)
      5  variance
      6  normalised position of min  (argmin / (len-1), or 0 if len==1)
      7  normalised position of max
      8  last minus first score
      9  score gap at min position   (score_before_min - score_at_min), or 0 at boundary
    """
    n = len(step_scores)
    if n == 0:
        return np.zeros(10, dtype=np.float32)

    arr = np.array(step_scores, dtype=np.float64)
    mean = float(arr.mean())
    mn = float(arr.min())
    mx = float(arr.max())
    last = float(arr[-1])
    length = float(n)
    var = float(arr.var())

    if n > 1:
        pos_min = float(arr.argmin()) / (n - 1)
        pos_max = float(arr.argmax()) / (n - 1)
    else:
        pos_min = 0.0
        pos_max = 0.0

    delta_last_first = last - float(arr[0])

    amin = int(arr.argmin())
    gap_at_min = float(arr[amin - 1] - arr[amin]) if amin > 0 else 0.0

    return np.array(
        [mean, mn, mx, last, length, var, pos_min, pos_max, delta_last_first, gap_at_min],
        dtype=np.float32,
    )
