# Q2-b : Apply exponential moving average using "HoltWinters" for forecasting

# Import Library 
library(TTR)
library(reshape2)   # melt
library(ggthemes)   # Graph Theme
library(ggplot2)    # Plot graph
library(zoo)        # Time into year/month


# Read in dataset kc_weather
milkProduction <- read.csv("milk-Production.csv", header=TRUE)
head(milkProduction)

# Milk Production into Time Series Obj
ts_milkProduction <- ts(milkProduction$Pounds_per_Cow, frequency=12, start=c(1962,1))

# Create Dataframe for original data
df_milkProduction <- data.frame(lbs_per_cow=coredata(ts_milkProduction), date=as.Date(as.yearmon(time(ts_milkProduction))))

# Flip the column orders
df_milkProduction <- df_milkProduction[c("date", "lbs_per_cow")]

# Exponential Smoothing : Holt Winters
ES_HoltWinter <- HoltWinters(ts_milkProduction)

# Forecast Milk Production for the next 12 months with a confidence level of 95%
forecast_milk <- predict(ES_HoltWinter, n.ahead = 12, prediction.interval = T, level = 0.95)

# Plot Holt Winter exponential moving average graph
plot(ES_HoltWinter, forecast_milk)

# melt SMA obj into date, value (milk production) and variables
meltdf <- melt(ES_HoltWinter, id="date")
values<-data.frame(time=as.Date(as.yearmon(time(forecast_milk))),  value_forecast=as.data.frame(forecast_milk)$fit,  
                       dev=as.data.frame(forecast_milk)$upr-as.data.frame(forecast_milk)$fit)
fitted_values<-data.frame(time=as.Date(as.yearmon(time(ES_HoltWinter$fitted))),  value_fitted=as.data.frame(ES_HoltWinter$fitted)$xhat)
actual_values<-data.frame(time=as.Date(as.yearmon(time(ES_HoltWinter$x))),  Actual=c(ES_HoltWinter$x))

# Put together acutal and fitted values by times
graphset<-merge(actual_values,  fitted_values,  by='time',  all=TRUE)
graphset<-merge(graphset,  values,  all=TRUE,  by='time')
graphset[is.na(graphset$dev),  ]$dev<-0
graphset$Fitted<-c(rep(NA,  NROW(graphset)-(NROW(values) + NROW(fitted_values))),  fitted_values$value_fitted,  values$value_forecast)

# Melt time, acutal and fitted value
graphset.melt<-melt(graphset[, c('time', 'Actual', 'Fitted')], id='time')

# Error area color
error.ribbon='gold'
line.size = 1
# Plot the line graph
ggplot(graphset.melt,  aes(x=time, y=value)) + geom_ribbon(data=graphset, aes(x=time, y=Fitted, ymin=Fitted-dev,  ymax=Fitted + dev),  alpha=.2,  fill=error.ribbon) +
  geom_line(aes(colour=variable), size=line.size) +  xlab('Time') + ylab('Milk Production lbs/cow') + scale_colour_brewer("Legend", palette = "Set1") +
  ggtitle("Holt Winters Actual v.s. Fitted") +  theme(plot.title = element_text(hjust = 0.5)) +
  scale_color_ptol() +  theme_minimal()

ggplot(graphset.melt,  aes(time, value, fill=variable)) + geom_bar(stat="identity")