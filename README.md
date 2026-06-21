# Air Raid Alert Forecasting in Ukraine

This project analyzes historical air raid alert data in Ukraine and predicts future alerts using machine learning. It predicts two main values:
1. The silence duration before the next alert starts.
2. The duration of that next alert once it starts.

The program runs as a chained forecast: it first predicts the silence duration, calculates the start time of the next alert, and then predicts the alert duration.

## Machine Learning Models

We use the Random Forest Regressor algorithm. The target variables are log-transformed during training to handle extreme values.

### Model 1: Silence Duration Prediction
* Goal: Predict the hours of peace (silence) after an alert ends.
* Features: Durations of the last two alerts and silences, the end hour, day of the week, month of the last alert, and the region (one-hot encoded).

### Model 2: Alert Duration Prediction
* Goal: Predict how long the next alert will last.
* Features: Durations of the last two alerts and silences (including the predicted silence from Model 1), the predicted start hour, day of the week, month of the alert, and the region (one-hot encoded).

### Metrics
We use Mean Absolute Error (MAE) calculated individually for each region. You can check the results in these plots:
* [Silence Prediction Error by Region](mae_silence_by_oblast.png)
* [Alert Duration Prediction Error by Region](mae_duration_by_oblast.png)

## How to Run

1. Create a virtual environment:
   python -m venv .venv

2. Activate the virtual environment:
   * Windows: .venv\Scripts\activate
   * Linux/macOS: source .venv/bin/activate

3. Install the dependencies:
   pip install -r requirements.txt

4. Run the interactive console application:
   python main.py

## Example Output

When you run the program and select a region (e.g., Kyiv), it shows:

```
Selected region: Kyiv

Latest historical state in the region:
  - Last alert: 0.62 hours (37 min)
  - Second to last alert: 0.40 hours (24 min)
  - Last silence: 5.95 hours (5 hours 57 min)
  - Second to last silence: 2.14 hours (2 hours 8 min)

Prediction results:
  1) Expected silence duration (before the next alert starts):
     -> 6.80 hours (6 hours 48 min) (error: +/- 12.67 hours)
  2) Predicted duration of the next alert:
     -> 0.55 hours (33 min) (error: +/- 0.81 hours)
```
