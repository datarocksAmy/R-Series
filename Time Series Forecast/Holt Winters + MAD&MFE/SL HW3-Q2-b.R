# Q2-b : Apply exponential moving average using "HoltWinters" for forecasting

# Import Library 
library(TTR)
library(reshape2)   # melt
library(ggthemes)   # Graph Theme
library(ggplot2)    # Plot graph
library(zoo)        # Time into year/month
library(stats)
library(outliers)


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
graphset[is.na(graphset)] <- 0

# Melt time, acutal and fitted value
graphset.melt<-melt(graphset[, c('time', 'Actual', 'Fitted')], id='time')

# Error area color
error.ribbon='gold'
line.size = 1

# Plot the line graph for fitted value v.s. actual value
ggplot(graphset.melt,  aes(x=time, y=value)) + geom_ribbon(data=graphset, aes(x=time, y=Fitted, ymin=Fitted-dev,  ymax=Fitted + dev),  alpha=.2,  fill=error.ribbon) +
  geom_line(aes(colour=variable), size=line.size) +  xlab('Time') + ylab('Milk Production lbs/cow') + scale_colour_brewer("Legend", palette = "Set1") +
  ggtitle("Holt Winters Actual v.s. Fitted") +  theme(plot.title = element_text(hjust = 0.5)) +
  scale_color_ptol() +  theme_minimal()

ggplot(graphset.melt,  aes(time, value, color=variable)) + geom_point()

# ----------------------------------------------------------------------------------------------------------------------
# Calculate Mean Absolute Diviation (MAD) & Mean Forecast Error (MFE)
# Median Fitted Value and actual values
fitted_mean <- mean(fitted_values$value_fitted)
actual_mean <- mean(actual_values$Actual)

# ABS value
fitted_abs <- abs(fitted_values$value_fitted - fitted_mean)
actual_abs <- abs(actual_values$Actual - actual_mean)

# Mean abs
fitted_abs_mean <- mean(fitted_abs)
actua_abs_mean <- mean(actual_abs)
# ---------------------------------------------------- MAD ----------------------------------------------------
# MAD!
fitted_MAD <- abs(graphset$value_fitted - fitted_mean)/ nrow(graphset)
actual_MAD <- abs(graphset$Actual - actual_mean) / nrow(graphset)

# MAD Dataframe
fitted_MAD_DF <- data.frame(graphset$time, fitted_MAD)
colnames(fitted_MAD_DF)[1] <- "time"

# Filter out outliers
F_fitted_MAD_DF <- subset(fitted_MAD_DF, fitted_MAD_DF$fitted_MAD < 4)

actual_MAD_DF <- data.frame(graphset$time, actual_MAD)
colnames(actual_MAD_DF)[1] <- "time"

# Filter out outliers
F_actual_MAD_DF <- subset(actual_MAD_DF, fitted_MAD_DF$fitted_MAD < 4)

# Put together fitted and actual data by time
combine_MAD <- merge(F_fitted_MAD_DF, F_actual_MAD_DF)

# Melt the data with label!
holtWinters_MAD <- melt(combine_MAD, id="time")

# Melt
fitted_melt <- melt(fitted_MAD_DF, id="fitted_values.time")
ggplot(fitted_melt, aes(x=fitted_values.time, y = value, color = fitted_values.time)) + geom_point(size = 2) + 
  theme_minimal() + scale_color_gradient(low = "#0091ff", high = "#f0650e")

actual_melt <- melt(actual_MAD_DF, id="actual_values.time")
ggplot(actual_melt, aes(x=actual_values.time, y = value, color = actual_values.time)) + geom_point(size = 2) + 
  theme_minimal() + scale_color_gradient(low = "#32aeff", high = "#f2aeff")

# fitted MAD v.s Actual MAD Bar Graph
ggplot(data=holtWinters_MAD, aes(x=time, y=value, fill=variable)) +  geom_bar(stat="identity", width=20, position=position_dodge(40)) +
scale_x_date(date_labels="%Y",date_breaks="1 year") + ggtitle("Holt Winters Actual v.s. Fitted - MAD") + 
  labs(x="Time(year)",y="Milk Production lbs/cow") + scale_fill_discrete(name="MAD", labels=c("Fitted","Actual"))

# MAD Difference 
MAD_dif <- data.frame(combine_MAD$time, (combine_MAD$actual_MAD-combine_MAD$fitted_MAD))
colnames(MAD_dif)[1] <- "time"
colnames(MAD_dif)[2] <- "MAD Diff"
# ---------------------------------------------------- MFE ----------------------------------------------------
# Mean Forecast Error

# MFE
fitted_MFE <- (graphset$value_fitted - fitted_mean)/ nrow(graphset)
actual_MFE <- (graphset$Actual - actual_mean) / nrow(graphset)

# MFE Dataframe
fitted_MFE_DF <- data.frame(graphset$time, fitted_MFE)
colnames(fitted_MFE_DF)[1] <- "time"

# Filter out outliers
F_fitted_MFE_DF <- subset(fitted_MFE_DF, abs(fitted_MFE_DF$fitted_MFE) < 4)

actual_MFE_DF <- data.frame(graphset$time, actual_MFE)
colnames(actual_MFE_DF)[1] <- "time"

# Filter out outliers
F_actual_MFE_DF <- subset(actual_MFE_DF, abs(fitted_MFE_DF$fitted_MFE) < 4)

# Put together fitted and actual data by time
combine_MFE <- merge(F_fitted_MFE_DF, F_actual_MFE_DF)

# Melt the data with label!
holtWinters_MFE <- melt(combine_MFE, id="time")

# fitted MFE v.s Actual MFE Bar Graph
ggplot(data=holtWinters_MFE, aes(x=time, y=value, fill=variable)) +  geom_bar(stat="identity", width=20, position=position_dodge(40)) +
  scale_x_date(date_labels="%Y",date_breaks="1 year") + ggtitle("Holt Winters Actual v.s. Fitted - MFE") + 
  labs(x="Time(year)",y="Milk Production lbs/cow") + scale_fill_discrete(name="MFE", labels=c("Fitted","Actual")) + 
  scale_fill_manual(values = c("tomato", "skyblue2"))

# MFE Difference 
MFE_dif <- data.frame(combine_MFE$time, (combine_MFE$actual_MFE-combine_MFE$fitted_MFE))
colnames(MFE_dif)[1] <- "time"
colnames(MFE_dif)[2] <- "MFE Diff"

# Merge two DF
MAD_MFE_dif_DF <- merge(MAD_dif, MFE_dif)

# Melt MAD & MFE difference
MAD_MFE_dif_melt <- melt(MAD_MFE_dif_DF, id = "time")

# Plot Line Chart
ggplot(MAD_MFE_dif_melt, aes(x=time, y = value, group=variable, color = factor(variable, labels = c("MFE", "MAD")))) +
  geom_line(size=1) + ggtitle("MAD & MFE Forecast Bias") + labs(x="Time",y="Milk Production lbs/cow", color="Forecast Error Method") +
  scale_color_fivethirtyeight() + theme_fivethirtyeight() + scale_x_date(date_labels="%Y",date_breaks  ="1 year") +
  theme(plot.title = element_text(hjust = 0.5)) + ylim(0.45, -0.45)