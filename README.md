# Ex-06: Exponential Smoothing Methods
## NAME : V S SREE VIVEKA
## REG NO : 2305001031
## DEPT : CSE
## Aim
To implement and compare Simple Exponential Smoothing, Holt's Linear Trend Method, and Holt-Winters Seasonal Method for time-series forecasting.

## Objectives
1. To understand exponential smoothing techniques.
2. To apply Simple Exponential Smoothing and observe the effect of changing the smoothing level.
3. To apply Holt's method to time-series data containing a trend.
4. To study the effect of a stronger trend on forecasting.
5. To apply Holt-Winters method to seasonal data.
6. To understand how seasonal periods affect forecasting.

## Activity 1: Change the Smoothing Level

### Program

```python

# Step 1: Import Required Libraries
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import SimpleExpSmoothing

# Step 2: Create Sales Data
sales = [100, 120, 110, 115, 105,
         125, 118, 122, 110, 119]
data = pd.Series(sales)

# Step 3: Create Simple Exponential Smoothing Model
model = SimpleExpSmoothing(
    data,
    initialization_method="estimated"
)

# Step 4: Fit Model with Smoothing Level 0.3
fit1 = model.fit(smoothing_level=0.3, optimized=False)
forecast1 = fit1.forecast(3)

print("Forecast with Smoothing Level 0.3:")
print(forecast1)

# Step 5: Fit Model with Smoothing Level 0.7
fit2 = model.fit(smoothing_level=0.7, optimized=False)
forecast2 = fit2.forecast(3)

print("\nForecast with Smoothing Level 0.7:")
print(forecast2)

# Step 6: Plot and Compare Results
plt.figure(figsize=(10, 5))
plt.plot(data.index, data, marker="o", label="Actual")
plt.plot(fit1.fittedvalues.index, fit1.fittedvalues,
         marker="x", label="Alpha 0.3")
plt.plot(fit2.fittedvalues.index, fit2.fittedvalues,
         marker="s", label="Alpha 0.7")

plt.title("Simple Exponential Smoothing Comparison")
plt.xlabel("Time")
plt.ylabel("Sales")
plt.legend()
plt.grid(True)
plt.show()

```

## Output

<img width="381" height="125" alt="image" src="https://github.com/user-attachments/assets/79ee8a95-4f11-4a58-964a-017e05d81438" />
<img width="377" height="142" alt="image" src="https://github.com/user-attachments/assets/c1ff50d5-80cd-4d58-b988-160d12a2c9e9" />
<img width="850" height="470" alt="image" src="https://github.com/user-attachments/assets/4947424d-dab1-4006-8d2f-7430bca762ff" />

## Activity 2: Observe a Stronger Trend

### Program

```python
# Step 1: Import Required Libraries
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import Holt

# Step 2: Create Two Sales Datasets
sales1 = [100, 105, 112, 118, 125,
          130, 138, 145, 152, 160]

sales2 = [100, 110, 125, 140, 160,
          175, 190, 210, 225, 245]

data1 = pd.Series(sales1, dtype=float)
data2 = pd.Series(sales2, dtype=float)

# Step 3: Apply Holt's Trend Method
model1 = Holt(data1, initialization_method="estimated")
model2 = Holt(data2, initialization_method="estimated")

# Step 4: Fit the Models and Forecast
fit1 = model1.fit(optimized=True)
fit2 = model2.fit(optimized=True)

forecast1 = fit1.forecast(3)
forecast2 = fit2.forecast(3)

print("Forecast for Dataset 1:")
print(forecast1)

print("\nForecast for Dataset 2:")
print(forecast2)

# Step 5: Plot and Compare Results
plt.figure(figsize=(10, 5))

plt.plot(data1.index, data1, marker="o",
         label="Original Sales")
plt.plot(data2.index, data2, marker="s",
         label="Stronger Trend Sales")

plt.title("Comparison of Sales Trends")
plt.xlabel("Time")
plt.ylabel("Sales")
plt.legend()
plt.grid(True)
plt.show()
```
## Output

<img width="257" height="267" alt="image" src="https://github.com/user-attachments/assets/6c473b36-0f26-4ac4-916a-40e94359ae42" />
<img width="850" height="470" alt="image" src="https://github.com/user-attachments/assets/933736f7-adfb-4054-858c-1eacc458ee8d" />


## Activity 3: Change Seasonal Periods to 12

### Program

```python
# Step 1: Import Required Libraries
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Step 2: Create Monthly Sales Data
seasonal_sales = [
    100, 130, 160, 120, 110, 145, 175, 130,
    120, 155, 190, 140, 110, 140, 170, 130,
    120, 155, 190, 145, 130, 165, 205, 150,
    130, 165, 200, 155, 140, 175, 215, 160,
    140, 175, 215, 165
]

dates = pd.date_range(start="2022-01-01",
                      periods=len(seasonal_sales), freq="MS")
data = pd.Series(seasonal_sales, index=dates)

print("Monthly Sales Data:")
print(data)

# Step 3: Create Holt-Winters Model
model = ExponentialSmoothing(
    data,
    trend="add",
    seasonal="add",
    seasonal_periods=12
)

# Step 4: Fit the Model
fit = model.fit()

# Step 5: Forecast the Next 12 Months
forecast = fit.forecast(12)

print("\nForecast for the Next 12 Months:")
print(forecast)

# Step 6: Plot Actual Data and Forecast
plt.figure(figsize=(12, 5))
plt.plot(data.index, data, marker="o", label="Actual Sales")
plt.plot(forecast.index, forecast, marker="x",
         linestyle="--", label="12-Month Forecast")
plt.title("Holt-Winters Monthly Seasonal Forecast")
plt.xlabel("Month")
plt.ylabel("Sales")
plt.legend()
plt.grid(True)
plt.show()
```
## Output
<img width="245" height="676" alt="image" src="https://github.com/user-attachments/assets/f1be57d9-68eb-4204-b7b8-77147d49a46e" />
<img width="362" height="327" alt="image" src="https://github.com/user-attachments/assets/6439a1c1-1d39-401e-8b80-4b4c7d064fb4" />
<img width="1010" height="470" alt="image" src="https://github.com/user-attachments/assets/463b2eae-d6e6-4d71-8d67-554cb537db1c" />


## Result
All three exponential smoothing activities were completed successfully. The effect of the smoothing level, the strength of the trend, and the seasonal period on time-series forecasting was studied using Python.
