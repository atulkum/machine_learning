# -*- coding: utf-8 -*-
"""
Created on Wed Aug  6 23:28:04 2014

@author: atulkumar
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

hourly = pd.read_csv("FremontHourly.csv", index_col='Date', parse_dates=True)
hourly.columns = ['northbound', 'southbound']
hourly['total'] = hourly['northbound'] + hourly['southbound']

daily = hourly.resample('d', 'sum')
weekly = daily.resample('w', 'sum')

weekly[['northbound', 'southbound', 'total']].plot()
plt.ylabel('Weekly riders');

pd.stats.moments.rolling_mean(daily['total'], 30).plot();


def hours_of_daylight(date, axis=23.44, latitude=47.61):
    """Compute the hours of daylight for the given date"""
    diff = date - pd.datetime(2000, 12, 21)
    day = diff.total_seconds() / 24. / 3600
    day %= 365.25
    m = 1. - np.tan(np.radians(latitude)) * np.tan(np.radians(axis) * np.cos(day * np.pi / 182.625))
    m = max(0, min(m, 2))
    return 24. * np.degrees(np.arccos(1 - m)) / 180.

# add this to our weekly data
weekly['daylight'] = list(map(hours_of_daylight, weekly.index))
daily['daylight'] = list(map(hours_of_daylight, daily.index))

weekly['daylight'].plot()
plt.ylabel('hours of daylight (Seattle)');

plt.scatter(weekly['daylight'], weekly['total'])
plt.xlabel('daylight hours')
plt.ylabel('weekly bicycle traffic');


from sklearn.linear_model import LinearRegression

X = weekly[['daylight']].to_dense()
y = weekly['total']
clf = LinearRegression(fit_intercept=True).fit(X, y)

weekly['daylight_trend'] = clf.predict(X)
weekly['daylight_corrected_total'] = weekly['total'] - weekly['daylight_trend'] + weekly['daylight_trend'].mean()

xfit = np.linspace(7, 17)
yfit = clf.predict(xfit[:, None])
plt.scatter(weekly['daylight'], weekly['total'])
plt.plot(xfit, yfit, '-k')
plt.title("Bicycle traffic through the year")
plt.xlabel('daylight hours')
plt.ylabel('weekly bicycle traffic');

print(clf.coef_[0])

trend = clf.predict(weekly[['daylight']].as_matrix())
plt.scatter(weekly['daylight'], weekly['total'] - trend + np.mean(trend))
plt.plot(xfit, np.mean(trend) + 0 * yfit, '-k')
plt.title("weekly traffic (detrended)")
plt.xlabel('daylight hours')
plt.ylabel('adjusted weekly count');

weekly[['total', 'daylight_trend']].plot()
plt.ylabel("total weekly riders");

weekly['daylight_corrected_total'].plot()
rms = np.std(weekly['daylight_corrected_total'])
plt.ylabel("adjusted total weekly riders")
print("root-mean-square about trend: {0:.0f} riders".format(rms))

days = ['Mon', 'Tues', 'Wed', 'Thurs', 'Fri', 'Sat', 'Sun']
daily['dayofweek'] = daily['total'].index.dayofweek

grouped = daily.groupby('dayofweek')['total'].mean()
grouped.index = days

grouped.plot()
plt.title("Average Traffic By Day")
plt.ylabel("Average Daily Crossings");

for i in range(7):
    daily[days[i]] = (daily.index.dayofweek == i).astype(float)

# de-trend on days of the week and daylight together
X = daily[days + ['daylight']]
y = daily['total']
clf = LinearRegression().fit(X, y)
daily['dayofweek_trend'] = clf.predict(X)
daily[['total', 'dayofweek_trend']].plot();

print(vars(clf))

daily['dayofweek_corrected'] = (daily['total'] - daily['dayofweek_trend'] + daily['dayofweek_trend'].mean())
print("rms = {0:.0f}".format(np.std(daily['dayofweek_corrected'])))
daily['dayofweek_corrected'].plot();

# Read the weather file
weather = pd.read_csv('SeaTacWeather.csv', index_col='DATE', parse_dates=True, usecols=[2, 3, 6, 7])

# temperatures are in 1/10 deg C; convert to F
weather['TMIN'] = 0.18 * weather['TMIN'] + 32
weather['TMAX'] = 0.18 * weather['TMAX'] + 32

# precip is in 1/10 mm; convert to inches
weather['PRCP'] /= 254

weather['TMIN'].resample('w', 'min').plot()
weather['TMAX'].resample('w', 'max').plot()
plt.ylabel('Weekly Temperature Extremes (F)');
plt.title("Temperature Extremes in Seattle");

weather['PRCP'].resample('w', 'sum').plot();
plt.ylabel('Weekly precipitation (in)')
plt.title("Precipitation in Seattle");

daily = daily.join(weather)


columns = days + ['daylight', 'TMIN', 'TMAX', 'PRCP']

X = daily[columns]
y = daily['total']
clf = LinearRegression().fit(X, y)
daily['overall_trend'] = clf.predict(X)

# Plot the overall trend
daily[['total', 'overall_trend']].plot()
plt.ylabel('Daily bicycle traffic');

daily['overall_corrected'] = daily['total'] - daily['overall_trend'] + daily['overall_trend'].mean()
print("rms = {0:.0f}".format(np.std(daily['overall_corrected'])))
daily['overall_corrected'].plot()
plt.ylabel('corrected daily bicycle traffic');

pd.stats.moments.rolling_mean(daily['overall_corrected'], 30).plot()
plt.ylabel('Corrected daily bicycle traffic')
plt.title('1-month Moving Window Average');

daily['daycount'] = np.arange(len(daily))

columns = days + ['daycount', 'daylight', 'TMIN', 'TMAX', 'PRCP']
X = daily[columns]
y = daily['total']
final_model = LinearRegression().fit(X, y)
daily['final_trend'] = final_model.predict(X)

daily['final_corrected'] = daily['total'] - daily['final_trend'] + daily['final_trend'].mean()


daily['total'].plot()
daily['final_trend'].plot();

daily['final_corrected'].plot()
plt.ylabel('corrected ridership')
print("rms = {0:.0f}".format(np.std(daily['final_corrected'])))

vy = np.sum((y - daily['final_trend']) ** 2) / len(y)
X2 = np.hstack([X, np.ones((X.shape[0], 1))])
C = vy * np.linalg.inv(np.dot(X2.T, X2))
var = C.diagonal()

ind = columns.index('PRCP')
slope = final_model.coef_[ind]
error = np.sqrt(var[ind])
print("{0:.0f} +/- {1:.0f} daily crossings lost per inch of rain".format(-slope, error))

ind1, ind2 = columns.index('TMIN'), columns.index('TMAX')
slope = final_model.coef_[ind1] + final_model.coef_[ind2]
error = np.sqrt(var[ind1] + var[ind2])
print('{0:.0f} +/- {1:.0f} riders per ten degrees Fahrenheit'.format(10 * slope, 10 * error))

ind = columns.index('daylight')
slope = final_model.coef_[ind]
error = np.sqrt(var[ind])
print("{0:.0f} +/- {1:.0f} daily crossings gained per hour of daylight".format(slope, error))

ind = columns.index('daycount')
slope = final_model.coef_[ind]
error = np.sqrt(var[ind])
print("{0:.2f} +/- {1:.2f} new riders per day".format(slope, error))
print("{0:.1f} +/- {1:.1f} new riders per week".format(7 * slope, 7 * error))
print("annual change: ({0:.0f} +/- {1:.0f})%".format(100 * 365 * slope / daily['total'].mean(),
                                                    100 * 365 * error / daily['total'].mean()))