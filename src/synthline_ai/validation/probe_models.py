from __future__ import annotations

import numpy as np

from synthline_ai.generation.base import GenerationResult


def extract_features(images: list[np.ndarray], masks: list[np.ndarray]) -> np.ndarray:
    """Extract simple feature vectors from image-mask pairs.

    Features per image:
    - Mean and std of RGB channels in defect region (6 features)
    - Mean and std of RGB channels in background region (6 features)
    - Defect area ratio (1 feature)
    - Defect bounding box aspect ratio (1 feature)
    - Mean gradient magnitude at defect boundary (1 feature)
    - Contrast between defect and background (1 feature)
    Total: 16 features

    Args:
        images: List of image arrays, shape (H, W, 3).
        masks: List of mask arrays, shape (H, W) or (H, W, 1).

    Returns:
        np.ndarray: Array of shape (N, 16) containing extracted features.
    """
    import cv2

    if len(images) != len(masks):
        raise ValueError("Number of images and masks must match")

    num_images = len(images)
    features_list = []

    for i in range(num_images):
        img = images[i]
        mask = masks[i]

        if mask.ndim == 3:
            mask = mask.squeeze(-1)

        is_defect = mask > 0
        is_bg = ~is_defect

        # 1. Defect region features (6)
        if is_defect.any():
            defect_pixels = img[is_defect]
            defect_mean = np.mean(defect_pixels, axis=0)
            defect_std = np.std(defect_pixels, axis=0)
        else:
            defect_mean = np.zeros(3)
            defect_std = np.zeros(3)

        # 2. Background region features (6)
        if is_bg.any():
            bg_pixels = img[is_bg]
            bg_mean = np.mean(bg_pixels, axis=0)
            bg_std = np.std(bg_pixels, axis=0)
        else:
            bg_mean = np.zeros(3)
            bg_std = np.zeros(3)

        # 3. Defect area ratio (1)
        area_ratio = np.sum(is_defect) / max(mask.size, 1)

        # 4. Defect bounding box aspect ratio (1)
        aspect_ratio = 0.0
        if is_defect.any():
            y_indices, x_indices = np.where(is_defect)
            h = int(np.max(y_indices)) - int(np.min(y_indices)) + 1
            w = int(np.max(x_indices)) - int(np.min(x_indices)) + 1
            aspect_ratio = float(w) / float(max(h, 1))

        # 5. Mean gradient magnitude at defect boundary (1)
        grad_mag_boundary = 0.0
        if is_defect.any():
            try:
                # Convert to grayscale if it's RGB
                gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.shape[-1] == 3 else img
                gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
                gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
                mag = np.sqrt(gx**2 + gy**2)

                # Morphological gradient to find boundary of mask
                mask_uint8 = (is_defect * 255).astype(np.uint8)
                kernel = np.ones((3, 3), np.uint8)
                boundary_mask = cv2.morphologyEx(mask_uint8, cv2.MORPH_GRADIENT, kernel) > 0

                if boundary_mask.any():
                    grad_mag_boundary = float(np.mean(mag[boundary_mask]))
            except Exception:
                pass

        # 6. Contrast between defect and background (1)
        contrast = 0.0
        if is_defect.any() and is_bg.any():
            defect_gray_mean = float(np.mean(defect_mean))
            bg_gray_mean = float(np.mean(bg_mean))
            contrast = float(abs(defect_gray_mean - bg_gray_mean))

        feat = np.concatenate(
            [
                defect_mean,
                defect_std,
                bg_mean,
                bg_std,
                [area_ratio, aspect_ratio, grad_mag_boundary, contrast],
            ]
        )
        features_list.append(feat)

    if not features_list:
        return np.zeros((0, 16))

    return np.stack(features_list)


def train_probe(
    synthetic_images: list[np.ndarray],
    synthetic_masks: list[np.ndarray],
    synthetic_labels: list[int],
    real_images: list[np.ndarray],
    real_masks: list[np.ndarray],
    real_labels: list[int],
    random_seed: int = 42,
) -> dict[str, object]:
    """Train a lightweight classifier on synthetic data, evaluate on real data.

    Uses sklearn's GradientBoostingClassifier (small, fast).
    Reports precision, recall, F1 on both synthetic (cross-val) and real (held-out).

    Args:
        synthetic_images: List of synthetic image arrays.
        synthetic_masks: List of synthetic mask arrays.
        synthetic_labels: List of synthetic labels (0/1).
        real_images: List of real image arrays.
        real_masks: List of real mask arrays.
        real_labels: List of real labels (0/1).
        random_seed: Seed for reproducibility.

    Returns:
        dict with:
        - 'synthetic_cv_accuracy': float (5-fold cross-val on synthetic)
        - 'real_precision': float
        - 'real_recall': float
        - 'real_f1': float
        - 'real_accuracy': float
        - 'feature_importances': dict[str, float] (top features)
        - 'sim_to_real_gap': float (synthetic_cv_accuracy - real_accuracy)
        - 'model_params': dict
    """
    try:
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
        from sklearn.model_selection import cross_val_score
    except ImportError:
        raise ImportError(
            "scikit-learn is required for probe models. "
            "Install with `pip install synthline-ai[probe]` or `pip install scikit-learn`"
        ) from None

    if len(synthetic_images) < 2:
        raise ValueError("Need at least 2 synthetic samples to train probe")

    X_train = extract_features(synthetic_images, synthetic_masks)
    y_train = np.array(synthetic_labels)

    X_test = extract_features(real_images, real_masks)
    y_test = np.array(real_labels)

    clf = GradientBoostingClassifier(n_estimators=50, random_state=random_seed)

    if len(set(synthetic_labels)) < 2:
        return {
            "probe_error": (
                "Synthetic data contains only 1 class; both normal and defect samples are needed."
            ),
            "synthetic_cv_accuracy": 1.0,
            "real_precision": 0.0,
            "real_recall": 0.0,
            "real_f1": 0.0,
            "real_accuracy": 0.0,
            "sim_to_real_gap": 1.0,
            "feature_importances": {},
            "model_params": clf.get_params(),
        }

    # Cross validation on synthetic data
    # Use max(2, min(5, n_samples)) for cv splits
    cv_splits = max(2, min(5, len(synthetic_labels) // 2))
    if cv_splits >= 2 and len(set(synthetic_labels)) > 1:
        cv_scores = cross_val_score(clf, X_train, y_train, cv=cv_splits)
        syn_cv_acc = float(np.mean(cv_scores))
    else:
        # Fallback if too few samples
        clf.fit(X_train, y_train)
        syn_cv_acc = float(accuracy_score(y_train, clf.predict(X_train)))

    # Train on full synthetic
    clf.fit(X_train, y_train)

    # Evaluate on real
    y_pred = clf.predict(X_test)

    real_acc = float(accuracy_score(y_test, y_pred))
    real_prec = float(precision_score(y_test, y_pred, zero_division=0.0))
    real_rec = float(recall_score(y_test, y_pred, zero_division=0.0))
    real_f1 = float(f1_score(y_test, y_pred, zero_division=0.0))

    feature_names = [
        "defect_r_mean",
        "defect_g_mean",
        "defect_b_mean",
        "defect_r_std",
        "defect_g_std",
        "defect_b_std",
        "bg_r_mean",
        "bg_g_mean",
        "bg_b_mean",
        "bg_r_std",
        "bg_g_std",
        "bg_b_std",
        "area_ratio",
        "aspect_ratio",
        "grad_mag_boundary",
        "contrast",
    ]

    importances = clf.feature_importances_
    feat_imp = {name: float(imp) for name, imp in zip(feature_names, importances, strict=False)}

    return {
        "synthetic_cv_accuracy": syn_cv_acc,
        "real_precision": real_prec,
        "real_recall": real_rec,
        "real_f1": real_f1,
        "real_accuracy": real_acc,
        "feature_importances": feat_imp,
        "sim_to_real_gap": float(syn_cv_acc - real_acc),
        "model_params": clf.get_params(),
    }


def evaluate_dataset_quality(
    results: list[GenerationResult],
    real_images: list[np.ndarray] | None = None,
    real_masks: list[np.ndarray] | None = None,
    real_labels: list[int] | None = None,
    random_seed: int = 42,
) -> dict[str, object]:
    """High-level quality evaluation combining diversity metrics and optional probe model.

    Always computes:
    - feature diversity (std of feature vectors across synthetic set)
    - intra-class variance

    If real data provided, also runs train_probe().

    Args:
        results: List of GenerationResult objects from generation output.
        real_images: Optional list of real images.
        real_masks: Optional list of real masks.
        real_labels: Optional list of real labels.
        random_seed: Seed for reproducibility.

    Returns:
        dict with all metrics.
    """
    if not results:
        return {"error": "No results provided for evaluation"}

    syn_images = []
    syn_masks = []
    syn_labels = []

    for r in results:
        if r.image is not None and r.mask is not None:
            syn_images.append(r.image)
            syn_masks.append(r.mask)

            # Simple heuristic for label: 1 if mask has positive pixels, else 0
            label = 1 if (r.mask > 0).any() else 0
            syn_labels.append(label)

    if len(syn_images) == 0:
        return {"error": "No valid image-mask pairs in results"}

    metrics: dict[str, object] = {}

    # Extract features for diversity analysis
    syn_features = extract_features(syn_images, syn_masks)

    # Feature diversity: mean standard deviation across all features
    metrics["feature_diversity"] = float(np.mean(np.std(syn_features, axis=0)))

    # Intra-class variance
    classes = np.unique(syn_labels)
    intra_class_vars = []
    for c in classes:
        mask = np.array(syn_labels) == c
        if np.sum(mask) > 1:
            class_feats = syn_features[mask]
            intra_class_vars.append(np.mean(np.var(class_feats, axis=0)))

    if intra_class_vars:
        metrics["intra_class_variance"] = float(np.mean(intra_class_vars))
    else:
        metrics["intra_class_variance"] = 0.0

    # Probe model evaluation (if real data is provided)
    if real_images is not None and real_masks is not None and real_labels is not None:
        if len(real_images) == len(real_masks) == len(real_labels) and len(real_images) > 0:
            if len(syn_images) >= 2:
                try:
                    probe_metrics = train_probe(
                        synthetic_images=syn_images,
                        synthetic_masks=syn_masks,
                        synthetic_labels=syn_labels,
                        real_images=real_images,
                        real_masks=real_masks,
                        real_labels=real_labels,
                        random_seed=random_seed,
                    )
                    metrics.update(probe_metrics)
                except ImportError as e:
                    metrics["probe_error"] = str(e)
            else:
                metrics["probe_error"] = "Insufficient synthetic data for probe training"
        else:
            metrics["probe_error"] = "Mismatched or empty real data lists"

    return metrics


def compare_synthetic_vs_real_baselines(
    synthetic_images: list[np.ndarray],
    synthetic_masks: list[np.ndarray],
    synthetic_labels: list[int],
    real_images: list[np.ndarray],
    real_masks: list[np.ndarray],
    real_labels: list[int],
    test_ratio: float = 0.4,
    random_seed: int = 42,
) -> dict[str, object]:
    """Compare performance across Real-only, Synthetic-only, and Combined (Real+Synthetic) models.

    Evaluates each trained baseline on a held-out test split of real images.
    Demonstrates whether synthetic visual-data generation provides tangible lift over a real-only
    baseline with limited defect samples.

    Args:
        synthetic_images: Synthetic image arrays.
        synthetic_masks: Synthetic binary masks.
        synthetic_labels: Synthetic binary labels (0=good, 1=defect).
        real_images: Real image arrays.
        real_masks: Real binary masks.
        real_labels: Real binary labels.
        test_ratio: Fraction of real images reserved for held-out evaluation.
        random_seed: Seed for reproducible dataset partitioning and probe fitting.

    Returns:
        dict containing metrics for real_only, synthetic_only, augmented, and calculated lift.
    """
    try:
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
        from sklearn.model_selection import train_test_split
    except ImportError:
        raise ImportError(
            "scikit-learn is required for baseline comparisons. "
            "Install with `pip install synthline-ai[probe]` or `pip install scikit-learn`"
        ) from None

    if len(real_images) < 4:
        raise ValueError("At least 4 real images are required for held-out baseline comparison.")
    if len(synthetic_images) == 0:
        raise ValueError("Synthetic dataset cannot be empty.")

    indices = np.arange(len(real_images))
    unique_labels, counts = np.unique(real_labels, return_counts=True)
    stratify = real_labels if len(unique_labels) > 1 and int(np.min(counts)) >= 2 else None

    train_idx, test_idx = train_test_split(
        indices,
        test_size=test_ratio,
        random_state=random_seed,
        stratify=stratify,
    )

    real_train_imgs = [real_images[i] for i in train_idx]
    real_train_masks = [real_masks[i] for i in train_idx]
    real_train_y = [real_labels[i] for i in train_idx]

    real_test_imgs = [real_images[i] for i in test_idx]
    real_test_masks = [real_masks[i] for i in test_idx]
    real_test_y = [real_labels[i] for i in test_idx]

    syn_X = extract_features(synthetic_images, synthetic_masks)
    real_train_X = extract_features(real_train_imgs, real_train_masks)
    real_test_X = extract_features(real_test_imgs, real_test_masks)

    def _eval_model(
        X_train: np.ndarray,
        y_train: list[int],
        X_test: np.ndarray,
        y_test: list[int],
    ) -> dict[str, float]:
        if len(np.unique(y_train)) < 2:
            single_cls = y_train[0]
            preds = [single_cls] * len(y_test)
        else:
            clf = GradientBoostingClassifier(random_state=random_seed, n_estimators=50)
            clf.fit(X_train, y_train)
            preds = clf.predict(X_test)
        return {
            "accuracy": float(accuracy_score(y_test, preds)),
            "precision": float(precision_score(y_test, preds, zero_division=0)),
            "recall": float(recall_score(y_test, preds, zero_division=0)),
            "f1": float(f1_score(y_test, preds, zero_division=0)),
            "train_samples": float(len(y_train)),
        }

    real_only = _eval_model(real_train_X, real_train_y, real_test_X, real_test_y)
    synthetic_only = _eval_model(syn_X, synthetic_labels, real_test_X, real_test_y)

    aug_X = np.vstack([real_train_X, syn_X])
    aug_y = real_train_y + synthetic_labels
    augmented = _eval_model(aug_X, aug_y, real_test_X, real_test_y)

    f1_lift = augmented["f1"] - real_only["f1"]
    acc_lift = augmented["accuracy"] - real_only["accuracy"]

    return {
        "real_only": real_only,
        "synthetic_only": synthetic_only,
        "augmented": augmented,
        "test_samples": len(real_test_y),
        "f1_lift": float(f1_lift),
        "accuracy_lift": float(acc_lift),
        "synthetic_standalone_relative_f1": float(
            synthetic_only["f1"] / (real_only["f1"] + 1e-6)
        ),
        "is_beneficial": bool(f1_lift >= 0 and acc_lift >= 0),
    }

