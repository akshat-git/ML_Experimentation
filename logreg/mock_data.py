import numpy as np


def make_binary_data(seed=0, size=60):
    rng = np.random.default_rng(seed)
    half = size // 2

    class_a = rng.normal(loc=(-1.8, -1.0), scale=0.65, size=(half, 2))
    class_b = rng.normal(loc=(1.8, 1.1), scale=0.65, size=(size - half, 2))

    X = np.vstack([class_a, class_b])
    y = np.array([0] * half + [1] * (size - half))

    order = rng.permutation(size)
    return X[order], y[order]


def make_admission_data(seed=1, size=80):
    rng = np.random.default_rng(seed)
    gpa = rng.uniform(2.0, 4.0, size)
    test_score = rng.uniform(40, 100, size)
    noise = rng.normal(0, 0.55, size)

    logits = -12.0 + 2.7 * gpa + 0.065 * test_score + noise
    y = (logits > 0).astype(int)
    X = np.column_stack([gpa, test_score])
    return X, y


def make_noisy_admission_data(seed=2, size=90):
    rng = np.random.default_rng(seed)
    gpa = rng.uniform(2.0, 4.0, size)
    test_score = rng.uniform(35, 100, size)
    activity = rng.uniform(0.0, 3.0, size)
    noise = rng.normal(0, 1.0, size)

    logits = -14.0 + 2.4 * gpa + 0.055 * test_score + 0.7 * activity + noise
    y = (logits > 0).astype(int)
    X = np.column_stack([gpa, test_score, activity])
    return X, y