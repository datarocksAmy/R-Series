# Q1-b-i : Using 366 entries from "kc_weather" to build models for LDA, QDA and KNN. 
#          Split the dataset into training and testing for over 100+ step size.

# Import Library
library(caret)
library(MASS)
library(e1071)
library(class)

# Read in dataset kc_weather
KCWeather = read.csv("kc_weather.csv", header=TRUE)
head(KCWeather)

# Seperate Date into Year, Month, Day
KCWeather$Year <- year(KCWeather$Date)
KCWeather$Month <- month(KCWeather$Date)
KCWeather$Day <- day(KCWeather$Date)

# Parse data randomly into 80% Training and 20% Testing
all <- (1:366)
TrainRandom<- sample(1:366, 290, replace=F) # Take 290 data randomly as Training
TestRandom <- setdiff(all, TrainRandom)     # Rest of the data as Testing

# Set step size 
set.seed(400)

# Train and Test Dataset
TrainSet <- KCWeather[TrainRandom,]
TestSet <- KCWeather[TestRandom,]

# Labels for Training and Testing 
TestEvents <- KCWeather$Events[TestRandom]
TrainEvents <- KCWeather$Events[TrainRandom]

# --------------------------------------------------------- LDA ---------------------------------------------------------
# Train LDA Model
LDA_Model <- lda(Events ~ Temp.F + Dew_Point.F + Humidity.percentage + Sea_Level_Press.in + Visibility.mi + Wind.mph + Precip.in + Year + Month + Day, data = TrainSet)

# Test LDA Model
LDA_Predict <- predict(LDA_Model, TestSet)

# Confusion Matrix
LDA_Class <- LDA_Predict$class
table(LDA_Class, TestEvents)

# Test Error Rate 
mean(LDA_Class == TestEvents)

# Confusion Matrix ( Accuracy, Precision, Recall )
LDA_Confusion <- confusionMatrix(data = LDA_Class, reference = TestEvents, mode = "prec_recall")

# Accuracy + Precision + Recall
LDA_Accuracy <- LDA_Confusion$overall["Accuracy"]
LDA_Precision <- LDA_Confusion$byClass[, "Precision"]
LDA_Recall <- LDA_Confusion$byClass[, "Recall"]

# --------------------------------------------------------- QDA ---------------------------------------------------------
# Train QDA Model
QDA_Model <- qda(Events ~ Temp.F + Dew_Point.F + Humidity.percentage + Sea_Level_Press.in + Visibility.mi + Wind.mph + Precip.in + Year + Month + Day, data = TrainSet)

# Test QDA Model
QDA_Predict <- predict(QDA_Model, TestSet)

# Confusion Matrix
QDA_Class <- QDA_Predict$class
table(QDA_Class, TestEvents)

# Test Error Rate 
mean(QDA_Class == TestEvents)

# Confusion Matrix ( Accuracy, Precision, Recall )
QDA_Confusion <- confusionMatrix(data = QDA_Class, reference = TestEvents, mode = "prec_recall")

# Accuracy + Precision + Recall
QDA_Accuracy <- QDA_Confusion$overall["Accuracy"]
QDA_Precision <- QDA_Confusion$byClass[, "Precision"]
QDA_Recall <- QDA_Confusion$byClass[, "Recall"]

# --------------------------------------------------------- KNN ---------------------------------------------------------
# Preprocessing
preprocess <- preProcess(x = TrainSet, method = c("center", "scale"))

# Train Control
TrainCTRL <- trainControl(method="repeatedcv", repeats = 10)

# Build KNN model using Training Data
KNN_Model <- train(Events ~ ., data = TrainSet, method = "knn", trControl = TrainCTRL, preProcess = c("center","scale"), tuneLength = 50)

# Predict Testing Data using the KNN Model
KNN_Predict <- predict(KNN_Model, newdata = TestSet)

# Confusion Matrix
KNN_Confusion <- confusionMatrix(KNN_Predict, TestEvents, mode = "prec_recall")

# Accuracy + Precision + Recall
KNN_Accuracy <- KNN_Confusion$overall["Accuracy"]
KNN_Precision <- KNN_Confusion$byClass[, "Precision"]
KNN_Recall <- KNN_Confusion$byClass[,"Recall"]

# --------------------------------------------------------- COMPARE MODELS ----------------------------------------------------------
# Accuracy of each model
Model <- c("LDA", "QDA", "KNN")
Accuracy <- c(LDA_Accuracy, QDA_Accuracy, KNN_Accuracy)
sum_model_Accuracy <- data.frame(Model, Accuracy)

# Precision/Recall for each Class in each model
# LDA 
LDA_P <- c(LDA_Precision)
LDA_R <- c(LDA_Recall)
# QDA 
QDA_P <- c(QDA_Precision)
QDA_R <- c(QDA_Recall)
# KNN
KNN_P <- c(KNN_Precision)
KNN_R <- c(KNN_Recall)

# Summary Precision/Recall
sum_Precision <- data.frame(LDA_P, QDA_P, KNN_P)
sum_Recall <- data.frame(LDA_R, QDA_R, KNN_R)

