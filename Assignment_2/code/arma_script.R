library(tseries)
library(TTR)

# SET THE PATH TO THE FOLDER 'Assignment_2'
setwd("/home/bill/github_repos/cda/Cyber-Data-Analytics/Assignment_2");

# Read the dataset
train_data <- read.csv('./data/training.csv',header = TRUE)
test_data <- read.csv('./data/testing.csv',header = TRUE)

train_ts <- ts(train_data$LIT101, frequency = 86400, start = 1)
plot.ts(train_ts)



t <- train_data[1:300000,]
tss <- ts(t$LIT101, frequency= 86400, start=1)
dev.new()
plot.ts(tss)

components <- decompose(tss)
plot(components)
library(forecast)
# Automated forecasting using an ARIMA model
fit <- auto.arima(tss)

timeseriesarima <- arima(tss, order=c(0,1,1)) # fit an ARIMA(0,1,1) model
timeseriesforecasts <- forecast.Arima(timeseriesarima, h=100000)

plot.forecast(timeseriesforecasts)
fit <- forecast.Arima(tss, h=5)
plot.forecast(fit)
# predict next 5 observations
forecast(fit, 5)
dev.new()
plot(forecast(fit, 5))
