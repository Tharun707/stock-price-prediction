# Preparation and verification

Prepared from the three user-supplied files on November 14, 2026. Original files on the Desktop were left unchanged.

Changes:

- Replaced the machine-specific model path with a path relative to `main.py`.
- Corrected the original app's swapped actual/predicted labels and added date axes and legends.
- Corrected test-data scaler fitting and inverse scaling in the app; documented the unverified original training scaler.
- Added single-symbol, data availability, history length, model file, and error handling.
- Cached downloads for one hour and cached model loading; closed Matplotlib figures after use.
- Cleared notebook outputs, execution counts, and code-cell metadata. Retained the original training/evaluation source and documented its limitations.
- Added dependency ranges, ignore rules, and student-friendly documentation. These ranges are not a verified environment lockfile.
- Included the supplied model unchanged. No retraining, performance numbers, license, or author identity was invented.

Verification: Python source syntax and notebook code syntax were checked without executing training. Notebook outputs were confirmed empty. The model archive passed a ZIP integrity check; architecture and metadata were read without loading weights into TensorFlow. The copied artifact's SHA-256 was compared with the original. Source text was checked for common credential markers and the original absolute path. This is a limited scan, not a guarantee that all possible secrets have been detected.

Full application execution was not verified because TensorFlow, Streamlit, and yfinance were unavailable in the local environment. Data download, Keras compatibility, dependency resolution, browser behavior, and predictive quality remain unverified. A successful syntax check does not prove those behaviors.
