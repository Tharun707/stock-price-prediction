# Stock Price Prediction

A student project that uses Python and an LSTM neural network to compare predicted stock prices with real historical prices. The interface is built with Streamlit, so you can enter a stock symbol and view the results in a browser.

The project includes a training notebook and a saved model. It is useful for learning how stock data, moving averages, data scaling, and sequence models work together.

> **Learning project:** The app compares prices from a past period. It does not predict future dates, provide trading signals, or guarantee accurate results. Do not use it to make investment decisions.

## What the app does

- Accepts one stock symbol, with `GOOG` as the default.
- Downloads historical data using `yfinance`.
- Displays the downloaded stock data in a table.
- Shows closing prices with 50-day, 100-day, and 200-day moving averages.
- Uses the supplied LSTM model to estimate closing prices for a historical test period.
- Displays original and predicted prices on the same chart.
- Handles missing data, short histories, a missing model file, and download/model errors with messages.

It does not include live trading, a brokerage connection, future-date forecasts, or a model trained separately for every symbol.

## Tools used

| Tool | Why it is used |
| --- | --- |
| Python | Main programming language. |
| Streamlit | Creates the browser interface. |
| TensorFlow and Keras | Load and run the LSTM model. |
| yfinance | Downloads historical market data. |
| pandas and NumPy | Prepare tables and numerical arrays. |
| scikit-learn | Scales closing prices with MinMaxScaler. |
| Matplotlib | Draws the price charts. |
| JupyterLab | Opens and runs the training notebook. |

## Project files

```text
stock-price-prediction/
├── main.py                         Streamlit application
├── data.ipynb                      Original training notebook; saved outputs cleared
├── Stock Predictions Model.keras   Supplied model, about 2.2 MB
├── requirements.txt                Application dependencies
├── requirements-dev.txt            Optional notebook dependencies
├── .gitignore                      Local files to keep out of GitHub
└── docs/
    ├── MODEL.md                    Model details and important limitations
    └── PREPARATION.md              Changes and verification results
```

The model is included, so you do not need to train it before trying the app. Keep its filename unchanged and keep it beside `main.py`.

## How it works

### 1. Download historical prices

The app requests data from **January 1, 2012 to December 31, 2022**. The end date is passed to yfinance as its exclusive end boundary. This is a fixed historical period, not the latest market data. The app explicitly requests adjusted prices with `auto_adjust=True`.

### 2. Split the data in date order

The earlier 80% of closing prices is used to fit the price scaler. The later 20% is used for the comparison chart. The order is preserved because this is time-series data.

### 3. Scale the prices

`MinMaxScaler` learns the price range from the earlier portion. The app then transforms the test inputs using that same scaler. It does not fit the scaler on the later test prices. Values outside the earlier range can fall outside 0–1.

### 4. Build 100-day inputs

Each input contains 100 previous trading observations. These are trading observations, not 100 calendar days. The model uses that sequence to estimate the next observed closing price.

For later test points, the input includes actual earlier test prices. This is a one-step historical comparison, not a simulation that repeatedly feeds its own predictions into future days.

### 5. Display the results

The predicted values are converted back to the price scale using `inverse_transform` and plotted against the actual dates and prices.

The original training scaler was not supplied. The app reconstructs one from downloaded data, so an exact match to the saved model's original preprocessing is not verified. See [model notes](docs/MODEL.md).

## Setup and run on Windows

Use a 64-bit Python environment supported by TensorFlow. **Python 3.11** is a starting choice for these instructions. The dependency ranges are proposed setup ranges, not a fully tested lockfile. See the official [TensorFlow installation guide](https://www.tensorflow.org/install/pip) for platform requirements.

### Step 1: Download the project

```powershell
git clone https://github.com/Tharun707/stock-price-prediction.git
cd stock-price-prediction
```

You can also use GitHub's **Code → Download ZIP**, extract the files, and open PowerShell in the extracted folder.

### Step 2: Create a virtual environment

```powershell
py -3.11 -m venv .venv
```

A virtual environment keeps this project's Python packages separate from other projects. The commands below use its Python directly, so activating it or changing PowerShell's script policy is not necessary.

### Step 3: Install the packages

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

TensorFlow is a large package, so the first installation may take time. Internet access is also needed whenever the app downloads market data.

### Step 4: Start the application

```powershell
.\.venv\Scripts\python.exe -m streamlit run main.py
```

Open the local address printed in the terminal, normally [http://localhost:8501](http://localhost:8501). Keep the terminal running. To stop the app, press **Ctrl+C**.

For macOS/Linux, create the environment with `python3 -m venv .venv` and use `.venv/bin/python` in place of `.\.venv\Scripts\python.exe`. TensorFlow availability depends on the operating system and processor.

## Using the app

1. Start the application and open its local page.
2. Keep `GOOG` for the example used by the training notebook, or enter one other valid stock symbol.
3. Read the historical data table.
4. Compare the closing-price line with the moving-average lines.
5. Scroll to **Original price vs predicted price** for the historical model comparison.

A moving average is the average of the most recent observations in a window. For example, MA50 averages 50 observations. The first part of a moving-average line is blank until enough observations exist.

Changing the symbol does not retrain the model. Results for another stock use the same supplied model and are only an experiment.

## Training notebook

To open `data.ipynb`:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m jupyterlab
```

Select the notebook and run its cells in order. It downloads `GOOG`, creates 100-observation windows, builds the network, and requests training for 50 epochs with a batch size of 32. Its final cell saves `Stock Predictions Model.keras`; running it from the project folder can overwrite the supplied model. Back up the original if you want to keep it.

**Important:** The notebook is retained as the original training example. Its evaluation cells refit the scaler on the test data and reverse scaling by multiplication rather than a full inverse transform. Those cells need correction before you use their plot to evaluate performance. The cleaned app corrects those two steps, but the notebook has not been retrained or presented as a verified experiment.

The notebook ends its data request at December 21, 2022, while the app ends at December 31, 2022. That difference, changes in downloaded prices, and the missing original scaler limit reproducibility.

## Model structure

The saved model configuration contains:

| Layer | Size or dropout rate |
| --- | --- |
| LSTM | 50 units |
| Dropout | 0.2 |
| LSTM | 60 units |
| Dropout | 0.3 |
| LSTM | 80 units |
| Dropout | 0.4 |
| LSTM | 120 units |
| Dropout | 0.5 |
| Dense output | 1 value |

An LSTM is a neural network layer that learns from sequences. Dropout is used during training to reduce reliance on individual units. The notebook uses ReLU activations in the LSTM layers, Adam as its optimizer, and mean squared error as its loss.

The file metadata records **Keras 3.13.2** and a save time of **February 1, 2026**. File metadata and notebook code do not prove the exact training history or accuracy of the weights.

## Checks and known limits

- Python syntax, notebook structure, and model archive/configuration were checked during preparation.
- TensorFlow, Streamlit, and yfinance were not installed in the preparation environment. A full model load, live download, and browser run have not been verified.
- The original model file is included unchanged. It has not been retrained.
- No verified accuracy, profit, MAE, RMSE, or R² result is claimed.
- Downloaded prices and adjustment methods can change. The original price snapshot and fitted scaler were not supplied.
- The model's future performance and behavior on other stocks have not been established.

## Common problems

| Problem | What to check |
| --- | --- |
| `py -3.11` is not found | Install Python 3.11 or select another Python version supported by your TensorFlow release. |
| A package is missing | Install requirements using the same virtual-environment Python used to start Streamlit. |
| TensorFlow will not install | Check Python version, 64-bit support, processor, and the official installation guide. |
| No prices appear | Check the symbol and internet connection. The data provider may also temporarily limit requests. |
| Model is missing | Keep `Stock Predictions Model.keras` beside `main.py`. |
| Model cannot load | Check TensorFlow/Keras compatibility; see the saved model metadata in the model notes. |
| Prices do not extend to today | The dates are fixed in `main.py`. Changing them also changes the comparison and reconstructed scaler. |
| Another stock looks inaccurate | The notebook uses GOOG; the app reuses that model for all entered symbols. |

## Ideas for future work

- Correct the notebook evaluation and train again using a documented data snapshot.
- Save the training scaler together with the model.
- Add a simple baseline and properly measured test metrics.
- Compare performance across time periods without using future information.
- Add tests for preprocessing and model input shape.

These are future improvements, not completed features.

## License and credits

No license or confirmed contributor list was supplied, so neither has been invented. Confirm ownership before adding a license. Historical data is obtained through yfinance; follow the data provider's terms for your use. The project is an educational demonstration and is not investment advice.
