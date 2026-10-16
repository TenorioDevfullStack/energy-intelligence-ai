"""Initial detector and event policies; tune only with training/validation data."""
ANOMALY_TRAIN_FRACTION = 0.60
MIN_RECALL = 0.80
THRESHOLD_GRID = tuple(float(value) for value in range(101))
MAX_EVENT_GAP_MINUTES = 15.0
MIN_EVENT_POINTS = 2
