# Q2-a : 3+ different values for window size with simple moving average (SMA) for forecasting.

# Import Library 
library(TTR)        # Simple Time Series
library(timeSeries) # Time Series 
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

# Simple Moving Average with 5 different window size
sma_1 <- SMA(ts_milkProduction, n = 5)
sma_2 <- SMA(ts_milkProduction, n = 10)
sma_3 <- SMA(ts_milkProduction, n = 15)
sma_4 <- SMA(ts_milkProduction, n = 20)
sma_5 <- SMA(ts_milkProduction, n = 25)

# Create Dataframe for all SMA
df_sma1 <- data.frame(lbs_per_cow1=coredata(sma_1), date=as.Date(as.yearmon(time(sma_1))))
df_sma1 <- df_sma1[c("date", "lbs_per_cow1")]
df_sma2 <- data.frame(lbs_per_cow2=coredata(sma_2), date=as.Date(as.yearmon(time(sma_2))))
df_sma2 <- df_sma2[c("date", "lbs_per_cow2")]
df_sma3 <- data.frame(lbs_per_cow3=coredata(sma_3), date=as.Date(as.yearmon(time(sma_3))))
df_sma3 <- df_sma3[c("date", "lbs_per_cow3")]
df_sma4 <- data.frame(lbs_per_cow4=coredata(sma_4), date=as.Date(as.yearmon(time(sma_4))))
df_sma4 <- df_sma4[c("date", "lbs_per_cow4")]
df_sma5 <- data.frame(lbs_per_cow5=coredata(sma_5), date=as.Date(as.yearmon(time(sma_5))))
df_sma5 <- df_sma5[c("date", "lbs_per_cow5")]

# Put all 5 sma data into 1 dataframe
all_sma_lbs <- data.frame(df_sma1$lbs_per_cow1, df_sma2$lbs_per_cow2, df_sma3$lbs_per_cow3, df_sma4$lbs_per_cow4, df_sma5$lbs_per_cow5)
# Labeld with time
sma_original_lbs <- data.frame(df_milkProduction, all_sma_lbs)

# Plot all SMA obj with the original Milk Production
# melt SMA obj into date, value (milk production) and variables
meltdf <- melt(sma_original_lbs, id="date")
# Plot Line Chart
ggplot(meltdf, aes(x=date, y = value, group=variable, color = factor(variable, labels = c("Original Milk Production", "SMA #1", "SMA#2", "SMA#3", "SMA#4", "SMA#5")))) +
  geom_line(size=1) + ggtitle("5 different Window Size of SMA v.s. Original") + labs(x="Time",y="Milk Production lbs/cow", color="Category") +
  theme_wsj()+ scale_colour_wsj("colors6") + scale_x_date(date_labels="%Y-%b",date_breaks  ="6 month") +
  theme(axis.text.x = element_text(angle = 90, hjust = 1), legend.direction="horizontal", plot.title = element_text(hjust = 0.5)) 
