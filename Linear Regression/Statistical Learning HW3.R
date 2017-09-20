# Import Library
library(ggplot2)
library(RColorBrewer)
library(gridExtra)
library(corrplot)

# Linear Regression for "Auto Dataset"
autoData = read.csv("Auto-rev.csv", header=TRUE)

# Summary of the datasets
head(autoData)

# Multi-variables Linear Regression
modelLR2 <- lm(mpg ~ cylinders + displacement + weight + acceleration + year + origin, data = autoData)
modelLR3 <- lm(mpg ~ weight + year + origin, data = autoData)
modelLR4 <- lm(mpg ~ cylinders + displacement + horsepower + weight + acceleration + year + origin, data = autoData)

# Statistical Summary
summary(modelLR2)
summary(modelLR3)
summary(modelLR4)


# Plot Multiple Predictor Graph
graph_Multiple <- ggplot(autoData, aes(x = cylinders + displacement + weight + acceleration + year + origin, y = mpg, color = mpg)) + geom_point(size = 3, shape = 18) + labs(title = "Multiple Linear Regression")+ theme_minimal() +  geom_smooth(method = "lm", color = "darkorange")
graph_filter <- ggplot(autoData, aes(x = weight + year + origin, y = mpg, color = mpg)) + geom_point(size = 3, shape = 18) + labs(title = "Filtered Multiple Linear Regression")+ theme_minimal() +  geom_smooth(method = "lm", color = "darkorange")

print(graph_Multiple + scale_colour_gradient(low="brown1",high="royalblue3"))
print(graph_filter + scale_colour_gradient(low="gold",high="forest green"))

# Correlation Matrix for all the Data
mpgData <- autoData[1]
cData <- autoData[2]
disData <- autoData[3]
hpData <- autoData[4]
wData <- autoData[5]
accData <- autoData[6]
yearData <- autoData[7]
originData <- autoData[8]

# Create a new dataframe for numerical data
dataF <- data.frame(mpgData, cData, disData, hpData, wData, accData, yearData, originData)

# Calculate the correlation
all <- cor(dataF)

# Plot the correlation matrix
corrplot(all, method = "shade", type = "lower")
title("Dataset Correlation Matrix", line = 0)

