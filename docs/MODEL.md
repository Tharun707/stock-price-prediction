# Model and evaluation notes

## Supplied artifact

`Stock Predictions Model.keras` is a ZIP-format Keras archive containing `metadata.json`, `config.json`, and `model.weights.h5`. It is about 2.2 MB. The original bytes are preserved. Its configuration matches the notebook's four LSTM layers (50, 60, 80, 120 units), dropout layers, and one-value Dense output. Metadata identifies Keras 3.13.2 and save time 2026-02-01@19:38:05.

The app loads the model with `compile=False` because it only needs inference, not an optimizer setup. Keras documents this parameter in its [model loading reference](https://keras.io/api/models/model_saving_apis/model_saving_and_loading/). Safe loading defaults are retained.

## What is known

The notebook uses GOOG closing prices, an ordered 80/20 split, a 100-observation input window, Adam, mean squared error, 50 requested epochs, and batch size 32. A code cell requesting 50 epochs does not establish that the supplied artifact completed that training or that it was generated from this exact data.

## What was missing

- Original fitted scaler and exact training price snapshot.
- A locked training environment.
- Verified holdout metrics and baseline results.
- A reproducibility seed and independently verified training record.

## Preparation corrections in main.py

The original app fitted a new scaler to the test period, which both used information from later prices and changed the scale from the training representation. It also converted scaled prices back using only multiplication, omitting the minimum-price offset. The prepared app fits only the earlier training portion and calls `inverse_transform` for predictions. Actual prices come directly from the held-out closing prices.

This fixes the calculations in the app but does not recover the exact original scaler. The app requests adjusted prices explicitly; the original download calls relied on library defaults. No model accuracy claim follows from these corrections. The notebook remains an original reference with its evaluation issues stated clearly in the README.

This is rolling one-step historical inference. Each input uses 100 actual past observations, including earlier actual test observations where applicable. It does not generate a multi-day future trajectory. Other tickers reuse the GOOG example model without retraining. Do not describe the final graph as a future forecast or investment signal.
