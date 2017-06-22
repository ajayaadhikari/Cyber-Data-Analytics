# install.packages('tseries')
library(tseries)
# install.packages('TTR')
library(TTR)
# install.packages('forecast')
library(forecast)
# install.packages('caret')
library(caret)


# SET THE PATH TO THE FOLDER 'Assignment_2'
setwd("/home/bill/github_repos/cda/Cyber-Data-Analytics/Assignment_2");

# Read the dataset (pre-processed datasets)
train_data <- read.csv('./data/training_arma.csv',header = TRUE)
test_data <- read.csv('./data/testing_arma.csv',header = TRUE)

# initialize dataframe with results / number of rows same as test set
results <- data.frame(init=numeric(nrow(test_data)))

# Start the process of prediction [for each signal] (except labels of course)
#three_signals = names(train_data[1:3])
for (signal in names(train_data)[1:length(train_data)-1]){
  print(signal)

#for (signal in three_signals){
  
#signal = 'MV101'
# convert data into time series
train_ts <- ts(as.numeric(unlist(train_data[signal])), start = 1)
test_ts <- ts(as.numeric(unlist(test_data[signal])), start =1)

# plot the training and test sets
#plot.ts(train_ts)
#plot.ts(test_ts) 

# Decomposing to better understand the signal's  components (not needed for the assignment)
# components <- decompose(train_ts)
# plot(components)

# Automated forecasting using an ARIMA model
# picks the model with the best AIC value
number_of_predictions = length(test_ts) #predict all values for testing set
fit <- auto.arima(train_ts, d=0)
#for (i in 1:10){
#fit <- arima(train_ts, c(i,0,0))
#AIC(fit)
#i
#}

timeseriesforecasts <- forecast.Arima(fit, h=number_of_predictions)

# plot the prediction
#dev.new()
#plot.forecast(timeseriesforecasts)

# Results
# mean values of the prediction
mean_forecasts <- timeseriesforecasts$mean

# Definition of the threshold. If the difference between the predicted values and the
# actual values is larger than the threshold, we consider the value as anomaly.
# The threshold is based on the std of the training set 
# (as they did in "Modeling Heterogeneous Time Series Dynamics to Profile Big Sensor Data in
# Complex Physical Systems)

# 1.9
thr <- 2 * sd(unlist(train_data[signal]))

# Computation of the Labels
predictions <- ((abs(mean_forecasts - unlist(test_data[signal])))>thr)*1
true_labels <- test_data$Normal.Attack
#print(confusionMatrix(predictions, true_labels))

results[signal] <- data.frame(predictions)
}

# delete initialization column
results$init <- NULL

# create majority labels
majority_labels <- ((rowSums(results==1) - rowSums(results==0))>0)*1

print(confusionMatrix(majority_labels, true_labels))


