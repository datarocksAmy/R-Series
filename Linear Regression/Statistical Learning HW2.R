# Import Library
library(ggplot2)
library(gridExtra)
library(ggthemes)
library(scales)

# Read in dataset as tale 
heading <- c("X", "Y")
secondData = read.table("Assignment_DataSet-A.txt", header = TRUE, row.names = 1, as.is = TRUE, col.names = c("X", "Y"))
names(secondData) <- heading

# Linear Regression Model
LRX_Y <- lm(Y ~ X, data = secondData)
trans_LR <- lm(log(Y) ~ log(X), data = secondData)

# Plot density for Linear and Transformed Linear Model 
ggplot() + geom_density(aes(residuals(LRX_Y)), color="darkblue", fill="lightblue", alpha = 0.4)
ggplot() + geom_density(aes(residuals(trans_LR)), color="tomato", fill="tomato", alpha = 0.4)

# Statistical Summary
summary(LRX_Y)
summary(trans_LR)

# Plot for relationship between X and Y
graph_data <- ggplot(secondData, aes(x = X, y = Y)) + geom_point(size = 3, shape = 18, colour = "darkorange") + labs(title = "X and Y Data Points")+ theme_minimal()
print(graph_data)

# Plot for original and transformed graph
graph_original <- ggplot(secondData, aes(x = X, y = Y)) + geom_point(size = 4, shape = "+", colour = "orange") +  stat_smooth(data = secondData, aes(x = X, y = Y), method = "lm", col = "coral2") + labs(title = "Relationship between X and Y") 
graph_trans <- ggplot(secondData, aes(x = log(X), y = log(Y))) + geom_point(size = 5.5, shape = "*", colour = "orange") +  geom_smooth(method="lm")+ labs(title = "Transformed Relationship between log(X) and log(Y)") 

# Put two graphs in one
grid.arrange(graph_original, graph_trans, ncol = 2)