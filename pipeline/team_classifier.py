"""Jersey color-based team classification using K-means clustering in HSV space.

Extracts the torso region from each player bounding box, converts to HSV,
and clusters dominant colors using K-means. The two largest clusters are
assigned as team_a and team_b; the smallest as 'other' (referee/goalkeeper).
"""

from collections import Counter, deque
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier

from utils.logger import get_logger

logger = get_logger(__name__)

# Team label constants
TEAM_A = "team_a"
TEAM_B = "team_b"
TEAM_OTHER = "other"


def hsv_to_color_name(hsv: np.ndarray) -> str:
    """Convert an HSV color array [H, S, V] to a human-readable color name.

    OpenCV HSV ranges: H=0-179, S=0-255, V=0-255.

    Args:
        hsv: 1D array with [hue, saturation, value].

    Returns:
        Color name string, e.g. 'red', 'white', 'dark blue'.
    """
    if hsv is None or len(hsv) < 3:
        return "unknown"
    
    h, s, v = float(hsv[0]), float(hsv[1]), float(hsv[2])

    # Low saturation → achromatic
    if s < 40:
        if v > 180:
            return "white"
        elif v < 80:
            return "black"
        else:
            return "gray"

    # Low value → very dark
    if v < 50:
        return "black"

    # Map hue to color name (OpenCV H: 0-179 = 0-360 degrees)
    # Red wraps around: 0-10 and 160-179
    if h <= 10 or h >= 160:
        return "red"
    elif h <= 25:
        return "orange"
    elif h <= 35:
        return "yellow"
    elif h <= 85:
        return "green"
    elif h <= 130:
        return "blue"
    elif h <= 160:
        return "purple"
    return "unknown"


def histogram_to_color_name(hist: np.ndarray) -> str:
    """Convert an 18-bin HSV histogram feature vector to a human-readable color name.

    Feature layout matches ``_extract_color_features``:
      bins  0-15 : 16-bin Hue histogram of chromatic pixels (normalized)
      bin  16    : white fraction
      bin  17    : black fraction

    Args:
        hist: 18-element float32 array that sums to ~1.0.

    Returns:
        Color name string, e.g. 'red', 'white', 'blue'.
    """
    if hist is None or len(hist) < 18:
        return "unknown"

    white_frac = float(hist[16])
    black_frac = float(hist[17])
    chromatic_mass = float(hist[:16].sum())

    # Dominant achromatic: pick whichever fraction is larger
    if white_frac > 0.4 and white_frac > chromatic_mass:
        return "white"
    if black_frac > 0.4 and black_frac > chromatic_mass:
        return "black"

    if chromatic_mass < 0.1:
        # Almost entirely achromatic — pick by relative fraction
        if white_frac >= black_frac:
            return "white" if white_frac > 0.15 else "gray"
        return "black"

    # Find the dominant hue bin (0-15 → spans H=0-180 in 16 equal steps of 11.25)
    peak_bin = int(np.argmax(hist[:16]))
    peak_hue = peak_bin * (180.0 / 16)  # approximate centre of peak bin in OpenCV H units

    if peak_hue <= 10 or peak_hue >= 160:
        return "red"
    elif peak_hue <= 25:
        return "orange"
    elif peak_hue <= 35:
        return "yellow"
    elif peak_hue <= 85:
        return "green"
    elif peak_hue <= 130:
        return "blue"
    elif peak_hue <= 160:
        return "purple"
    return "red"  # wraps back to red


class TeamClassifier:
    """Classifies detected players into teams based on jersey color.

    Uses K-means clustering on HSV color features extracted from
    the torso region of each player's bounding box.

    Attributes:
        n_clusters: Number of clusters (typically 3: team_a, team_b, other).
        sample_region: Which part of the bounding box to sample ('torso' or 'full').
    """

    def __init__(
        self,
        n_clusters: int = 4,
        color_space: str = "HSV",
        sample_region: str = "torso",
        bootstrap_frames: int = 25,
        voting_window: int = 15,
    ):
        """Initialize the team classifier.

        Args:
            n_clusters: Number of K-means clusters (4 recommended: team_a, team_b,
                        goalkeeper, referee/other).
            color_space: Color space for feature extraction (only HSV supported).
            sample_region: 'torso' crops to middle 30% of bbox; 'full' uses entire bbox.
            bootstrap_frames: Frames to collect before running batch K-Means and
                              freezing the classifier. Higher values → more stable
                              cluster assignments at the cost of a short learning delay.
            voting_window: Per-track rolling history length for majority-vote smoothing.
        """
        self.n_clusters = n_clusters
        self.color_space = color_space
        self.sample_region = sample_region

        # Bootstrap state ─ accumulate features for N frames then freeze
        self.bootstrap_frames = bootstrap_frames
        self._bootstrap_features: List[np.ndarray] = []
        self._bootstrap_frames_seen: int = 0
        self._bootstrap_complete: bool = False
        self._frozen_knn: Optional[KNeighborsClassifier] = None

        # Per-track temporal smoothing
        self._voting_window = voting_window
        self._label_history: Dict[int, deque] = {}

        # Shared metadata used by get_team_colors()
        self._kmeans: Optional[KMeans] = None
        self._cluster_to_team: Dict[int, str] = {}
        self._cluster_centers: Optional[np.ndarray] = None

        logger.info(
            f"TeamClassifier initialized: n_clusters={n_clusters}, "
            f"bootstrap_frames={bootstrap_frames}, voting_window={voting_window}"
        )

    def _extract_color_features(
        self, frame: np.ndarray, bbox: np.ndarray
    ) -> Optional[np.ndarray]:
        """Extract an 18-dimensional HSV histogram feature from a player's torso crop.

        Feature layout (all bins normalized, vector sums to 1):
          bins  0-15 : 16-bin Hue histogram of chromatic (S≥40) non-green pixels
          bin  16    : fraction of white pixels  (S<40, V>150)
          bin  17    : fraction of black pixels  (S<40, V<80)

        This is substantially more discriminative than a single dominant-color Lab
        vector because it captures the *distribution* shape:
          - Red jersey   → spikes at bins 0 and 15 (hue wraps at 0/179)
          - White jersey → spike at bin 16, flat hue histogram
          - Yellow ref   → spike at bin ~4 (H≈30)
          - Blue jersey  → spike at bins 12-13 (H≈110-130)

        Args:
            frame: Full BGR frame.
            bbox: Bounding box as [x1, y1, x2, y2].

        Returns:
            18D float32 feature vector, or None if the crop is too small.
        """
        x1, y1, x2, y2 = map(int, bbox)

        # Clamp to frame boundaries
        h, w = frame.shape[:2]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        if x2 - x1 < 5 or y2 - y1 < 5:
            return None

        # Skip very small detections — torso crop would cover noise pixels only
        if (y2 - y1) < 40:
            return None

        if self.sample_region == "torso":
            # Crop to rows 25%–55% of bbox height (tighter torso: avoids face & legs)
            box_h = y2 - y1
            torso_top = y1 + int(box_h * 0.25)
            torso_bottom = y1 + int(box_h * 0.55)
            crop = frame[torso_top:torso_bottom, x1:x2]
        else:
            crop = frame[y1:y2, x1:x2]

        if crop.size == 0:
            return None

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

        # --- Mask out green pitch pixels ---
        # Must match detector.py green mask thresholds exactly
        lower_green = np.array([30, 40, 40])
        upper_green = np.array([85, 255, 255])
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        player_mask = cv2.bitwise_not(green_mask)
        pixels = hsv[player_mask > 0]  # shape (N, 3): H, S, V

        # Fallback: if masking removed everything (rare: green kit / dark frame)
        if len(pixels) < 4:
            pixels = hsv.reshape(-1, 3)
        if len(pixels) == 0:
            return None

        hue = pixels[:, 0].astype(np.float32)        # 0–179
        sat = pixels[:, 1].astype(np.float32)        # 0–255
        val = pixels[:, 2].astype(np.float32)        # 0–255

        # Partition into chromatic vs achromatic pixels
        chromatic_mask = sat >= 40
        white_mask = (sat < 40) & (val > 150)
        black_mask = (sat < 40) & (val < 80)

        # 16-bin Hue histogram over chromatic pixels only
        hue_chromatic = hue[chromatic_mask]
        if len(hue_chromatic) > 0:
            hue_hist, _ = np.histogram(hue_chromatic, bins=16, range=(0.0, 180.0))
        else:
            hue_hist = np.zeros(16, dtype=np.float32)

        # Build the 18-bin feature vector
        feature = np.zeros(18, dtype=np.float32)
        feature[:16] = hue_hist.astype(np.float32)
        feature[16] = float(np.sum(white_mask))
        feature[17] = float(np.sum(black_mask))

        # L1-normalize so the vector sums to 1 — makes Euclidean distance
        # in histogram space equivalent to histogram intersection distance
        total = feature.sum()
        if total > 0:
            feature /= total

        return feature

    # ------------------------------------------------------------------
    # Bootstrap helpers
    # ------------------------------------------------------------------

    def _freeze_bootstrap(self) -> None:
        """Run one batch K-Means on all accumulated bootstrap features and freeze
        a KNN classifier based on the resulting cluster centers.

        Called automatically once ``bootstrap_frames`` frames have been seen.
        After this point ``classify()`` switches from returning TEAM_OTHER to
        using the stable frozen KNN for every subsequent frame.
        """
        all_features = np.array(self._bootstrap_features)
        if len(all_features) < self.n_clusters:
            # Extend the bootstrap window if we haven't seen enough players yet
            logger.warning(
                f"Bootstrap: only {len(all_features)} features (need {self.n_clusters}), "
                f"extending collection by 10 frames…"
            )
            self.bootstrap_frames += 10
            return

        logger.info(
            f"Bootstrap: running batch K-Means on {len(all_features)} features "
            f"from {self._bootstrap_frames_seen} frames…"
        )

        kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=42,
            n_init=10,
            max_iter=300,
        )
        cluster_labels = kmeans.fit_predict(all_features)

        # Map clusters → teams via max-distance pair selection.
        # Teams choose kits to be maximally distinct from EACH OTHER and from the
        # referee. The pair of cluster centers with the LARGEST histogram-space
        # distance is therefore always the two teams — the referee's yellow/orange
        # sits geometrically between them and will not be selected, regardless of
        # how many bootstrap frames the referee appears in.
        cluster_sizes = {
            label: int(np.sum(cluster_labels == label))
            for label in range(self.n_clusters)
        }
        centers = kmeans.cluster_centers_
        best_pair = (0, 1)
        best_dist = -1.0
        for i in range(self.n_clusters):
            for j in range(i + 1, self.n_clusters):
                d = float(np.linalg.norm(centers[i] - centers[j]))
                if d > best_dist:
                    best_dist = d
                    best_pair = (i, j)

        # Within the winning pair label the larger cluster team_a for consistency
        a, b = best_pair
        if cluster_sizes[a] >= cluster_sizes[b]:
            cluster_to_team: Dict[int, str] = {a: TEAM_A, b: TEAM_B}
        else:
            cluster_to_team = {a: TEAM_B, b: TEAM_A}

        # Cluster merging: assign orphan clusters to nearest team if close enough.
        # Sun/shade can split one team into 2 K-means clusters. Without merging,
        # half of that team's players would be classified as "other".
        for c in range(self.n_clusters):
            if c in best_pair:
                continue
            dist_to_a = float(np.linalg.norm(centers[c] - centers[a]))
            dist_to_b = float(np.linalg.norm(centers[c] - centers[b]))
            nearest_team_dist = min(dist_to_a, dist_to_b)
            if nearest_team_dist < best_dist * 0.5:
                # Close enough to a team — merge into it
                merge_target = cluster_to_team[a] if dist_to_a < dist_to_b else cluster_to_team[b]
                cluster_to_team[c] = merge_target
                logger.info(
                    f"Bootstrap: merging cluster {c} into {merge_target} "
                    f"(dist={nearest_team_dist:.4f}, threshold={best_dist * 0.5:.4f})"
                )
            else:
                cluster_to_team[c] = TEAM_OTHER

        logger.info(
            f"Bootstrap max-distance pair: clusters {best_pair}, "
            f"histogram distance={best_dist:.4f}, sizes={cluster_sizes}"
        )

        self._cluster_to_team = cluster_to_team
        self._cluster_centers = kmeans.cluster_centers_
        self._kmeans = kmeans

        # Build frozen KNN: one training point per cluster center (nearest-centroid)
        all_cids = list(range(self.n_clusters))
        X_train = np.array([kmeans.cluster_centers_[cid] for cid in all_cids])
        y_train = np.array([cluster_to_team[cid] for cid in all_cids])
        self._frozen_knn = KNeighborsClassifier(n_neighbors=1)
        self._frozen_knn.fit(X_train, y_train)

        self._bootstrap_complete = True
        logger.info(
            f"Bootstrap frozen: mapping={cluster_to_team}, sizes={cluster_sizes}"
        )

    def _apply_voting(
        self,
        track_ids: Optional[np.ndarray],
        labels: List[str],
    ) -> List[str]:
        """Apply per-track majority-vote smoothing to raw per-frame labels.

        Maintains a rolling deque of recent labels for each track ID and returns
        the majority vote, damping single-frame flicker from occlusion or lighting.

        Args:
            track_ids: Integer tracker IDs aligned with ``labels``.
                       Pass ``None`` to skip smoothing (labels returned as-is).
            labels: Raw per-detection team labels.

        Returns:
            Smoothed label list of the same length as ``labels``.
        """
        if track_ids is None:
            return labels

        smoothed = list(labels)
        for i, (track_id, label) in enumerate(zip(track_ids, labels)):
            if track_id is None:
                continue
            tid = int(track_id)
            if tid not in self._label_history:
                self._label_history[tid] = deque(maxlen=self._voting_window)
            self._label_history[tid].append(label)
            # Only smooth once we have at least 3 observations for this track
            hist = self._label_history[tid]
            if len(hist) >= 3:
                smoothed[i] = Counter(hist).most_common(1)[0][0]

        return smoothed

    # ------------------------------------------------------------------
    # Main classification entry point
    # ------------------------------------------------------------------

    def classify(
        self,
        frame: np.ndarray,
        bboxes: np.ndarray,
        team_seeds: Optional[Dict[str, np.ndarray]] = None,
        track_ids: Optional[np.ndarray] = None,
    ) -> List[str]:
        """Classify a list of detected players into teams.

        Behavior depends on mode:

        * **Supervised** (``team_seeds`` provided): fits a KNN on UI-clicked crops
          and classifies directly — no bootstrap delay.
        * **Unsupervised** (default): accumulates color features for
          ``bootstrap_frames`` frames, runs one batch K-Means, then uses a frozen
          KNN for all subsequent frames.  Returns ``TEAM_OTHER`` for every
          detection during the learning phase.

        Per-track majority-vote smoothing is applied when ``track_ids`` is given,
        dampening label flicker across frames.

        Args:
            frame: Full BGR frame.
            bboxes: Bounding boxes, shape (N, 4) as [x1, y1, x2, y2].
            team_seeds: Optional dict mapping team label → Lab feature array(s)
                        from UI clicks.
            track_ids: Optional int array of ByteTrack IDs aligned with bboxes.

        Returns:
            List of team labels per detection: 'team_a', 'team_b', or 'other'.
        """
        if len(bboxes) == 0:
            return []

        # Extract color features for all detections
        features: List[np.ndarray] = []
        valid_indices: List[int] = []
        for i, bbox in enumerate(bboxes):
            feat = self._extract_color_features(frame, bbox)
            if feat is not None:
                features.append(feat)
                valid_indices.append(i)

        if len(features) < 2:
            # Too few valid crops — not worth classifying
            return [TEAM_OTHER] * len(bboxes)

        features_array = np.array(features)

        # --- SUPERVISED PATH (UI seeds provided) -----------------------
        if team_seeds and any(len(seeds) > 0 for seeds in team_seeds.values()):
            X_train: List[np.ndarray] = []
            y_train: List[str] = []
            for team_label, seeds in team_seeds.items():
                if seeds.ndim == 1:
                    seeds = seeds.reshape(1, -1)
                for s in seeds:
                    X_train.append(s)
                    y_train.append(team_label)

            if len(X_train) == 0:
                return [TEAM_OTHER] * len(bboxes)

            n_neighbors = min(3, len(X_train))
            knn = KNeighborsClassifier(n_neighbors=n_neighbors)
            knn.fit(np.array(X_train), np.array(y_train))
            pred_labels = knn.predict(features_array)

            # Update color-reporting metadata
            unique_teams = list(set(y_train))
            cluster_to_team = {}
            centers = []
            for i, team in enumerate(unique_teams):
                cluster_to_team[i] = team
                team_data = np.array([x for x, y in zip(X_train, y_train) if y == team])
                centers.append(np.mean(team_data, axis=0))
            self._kmeans = None
            self._cluster_to_team = cluster_to_team
            self._cluster_centers = np.array(centers)

            result = [TEAM_OTHER] * len(bboxes)
            for idx, pred in zip(valid_indices, pred_labels):
                result[idx] = pred
            return self._apply_voting(track_ids, result)

        # --- UNSUPERVISED PATH -----------------------------------------
        # Bootstrap phase: accumulate features until we have enough data
        if not self._bootstrap_complete:
            self._bootstrap_features.extend(features)
            self._bootstrap_frames_seen += 1

            if self._bootstrap_frames_seen >= self.bootstrap_frames:
                self._freeze_bootstrap()
            else:
                # Still collecting — return TEAM_OTHER during the learning phase
                return [TEAM_OTHER] * len(bboxes)

        # Post-bootstrap: classify with frozen KNN (no re-clustering)
        distances, _ = self._frozen_knn.kneighbors(features_array)
        pred_labels = self._frozen_knn.predict(features_array)

        # Confidence gate: if a player's nearest-centroid distance exceeds the
        # gate threshold, their color is too ambiguous to assign to a team —
        # override to "other". Uses min(50% of team separation, 0.8) to prevent
        # over-rejection when teams have similar colors (small separation).
        _CONFIDENCE_GATE_MAX = 0.8
        team_cids = [
            cid for cid, lbl in self._cluster_to_team.items()
            if lbl in (TEAM_A, TEAM_B)
        ]
        dist_threshold: Optional[float] = None
        if len(team_cids) == 2:
            team_sep = float(np.linalg.norm(
                self._cluster_centers[team_cids[0]] - self._cluster_centers[team_cids[1]]
            ))
            dist_threshold = min(team_sep * 0.50, _CONFIDENCE_GATE_MAX)

        result = [TEAM_OTHER] * len(bboxes)
        for idx, pred, dist in zip(valid_indices, pred_labels, distances.flatten()):
            if (
                dist_threshold is not None
                and pred in (TEAM_A, TEAM_B)
                and dist > dist_threshold
            ):
                result[idx] = TEAM_OTHER
            else:
                result[idx] = pred
        return self._apply_voting(track_ids, result)

    def get_team_colors(self) -> Dict[str, str]:
        """Return the dominant jersey color name for each team label.

        Should be called after at least one call to classify().

        Returns:
            Dict mapping team label to color name.
        """
        if self._cluster_centers is None or not self._cluster_to_team:
            return {}

        colors: Dict[str, str] = {}
        for cluster_id, team_label in self._cluster_to_team.items():
            if cluster_id < len(self._cluster_centers):
                center = self._cluster_centers[cluster_id]
                # Cluster centers are 18D histogram vectors — decode color directly
                # from the histogram rather than converting back through Lab→BGR→HSV
                colors[team_label] = histogram_to_color_name(center)

        return colors
