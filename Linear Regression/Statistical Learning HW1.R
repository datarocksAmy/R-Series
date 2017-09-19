# Import Library
library(ggplot2)

# Linear Regression for "Auto Dataset"
autoData = read.csv("Auto-rev.csv", header=TRUE)

# Summary of the datasets
head(autoData)

# Simple Plotting
#plot(mpg ~ horsepower, pch=8, data = autoData, col="orange")
#title("Relationship between mpg and horsepower")
#legend("topright", bty = "n",
#       lwd = 2, cex = 1.2, c("LSE", "Data Points"), col=c("blue", "orange"), pch=c(NA,8))

# Linear Regression
modelLR1 <- lm(mpg ~ horsepower, data = autoData)
# LR Line
abline(autoData.modelLR1, col="blue") 
# Linear Regression Results
summary(autoData.modelLR1)

# Pearson Correlation - mpg and horsepower
cor.test(~ mpg + horsepower, data=autoData, method = "pearson", conf.level = 0.95)

# Predict mpg on a horsepower of 98
df_mpg <- data.frame(horsepower = 98)
predict(modelLR1, df_mpg)
predict(modelLR1, df_mpg, interval = 'confidence')

# Plot for relationship between mpg and horsepower
LMGraph <- ggplot(autoData, aes(x= horsepower, y=mpg)) + geom_point(colour = "tomato") + theme_economist() + stat_smooth(method = "lm", col = "blue4")
print(LMGraph + labs(title="Relationship between mpg and horsepower", y="MPG", x="HORSEPOWER" + theme(plot.title = element_text(hjust = 0.5))))
