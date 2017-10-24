# Q1-b-i : Using 366 entries from "kc_weather" to build models for LDA, QDA and KNN. 
#          Split the dataset into training and testing for over 100+ iterations.

# Import Library
library(lubridate) # Extract Date to Year, Month, Day
library(caret)
library(MASS) # LDA
library(e1071)
library(class) # KNN
library(dplyr) # Row Name to Single Column
library(ggplot2)
library(gridExtra)

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
#LDA_Model <- lda(Events ~ Temp.F + Dew_Point.F + Humidity.percentage + Sea_Level_Press.in + Visibility.mi + Wind.mph + Precip.in + Year + Month + Day, data = TrainSet)
LDA_Model <- train(Events ~ Temp.F + Dew_Point.F + Humidity.percentage + Sea_Level_Press.in + Visibility.mi + Wind.mph + Precip.in + Year + Month + Day, data = TrainSet, method = 'lda', repeats=100)

# Test LDA Model
LDA_Predict <- predict(LDA_Model, TestSet)

# Confusion Matrix
#LDA_Class <- LDA_Predict$class  # When run lda() command
table(LDA_Predict, TestEvents)

# Test Error Rate 
mean(LDA_Predict == TestEvents)

# Confusion Matrix ( Accuracy, Precision, Recall )
LDA_Confusion <- confusionMatrix(data = LDA_Predict, reference = TestEvents, mode = "prec_recall")

# Accuracy + Precision + Recall
LDA_Accuracy <- LDA_Confusion$overall["Accuracy"]
LDA_Precision <- LDA_Confusion$byClass[, "Precision"]
LDA_Recall <- LDA_Confusion$byClass[, "Recall"]

# --------------------------------------------------------- QDA ---------------------------------------------------------
# Train QDA Model
#QDA_Model <- qda(Events ~ Temp.F + Dew_Point.F + Humidity.percentage + Sea_Level_Press.in + Visibility.mi + Wind.mph + Precip.in + Year + Month + Day, data = TrainSet)
QDA_Model <- train(Events ~ Temp.F + Dew_Point.F + Humidity.percentage + Sea_Level_Press.in + Visibility.mi + Wind.mph + Precip.in + Year + Month + Day, data = TrainSet, method = 'qda', repeats=100)

# Test QDA Model
QDA_Predict <- predict(QDA_Model, TestSet)

# Confusion Matrix
#QDA_Class <- QDA_Predict$class  # For qda() command
table(QDA_Predict, TestEvents)

# Test Error Rate 
mean(QDA_Predict == TestEvents)

# Confusion Matrix ( Accuracy, Precision, Recall )
QDA_Confusion <- confusionMatrix(data = QDA_Predict, reference = TestEvents, mode = "prec_recall")

# Accuracy + Precision + Recall
QDA_Accuracy <- QDA_Confusion$overall["Accuracy"]
QDA_Precision <- QDA_Confusion$byClass[, "Precision"]
QDA_Recall <- QDA_Confusion$byClass[, "Recall"]

# --------------------------------------------------------- KNN ---------------------------------------------------------
# Preprocessing
preprocess <- preProcess(x = TrainSet, method = c("center", "scale"))

# Train Control
TrainCTRL <- trainControl(method="repeatedcv", repeats = 100)

# Build KNN model using Training Data
KNN_Model <- train(Events ~ Temp.F + Dew_Point.F + Humidity.percentage + Sea_Level_Press.in + Visibility.mi + Wind.mph + Precip.in + Year + Month + Day, data = TrainSet, method = "knn", trControl = TrainCTRL, preProcess = c("center","scale"), tuneLength = 20)

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
sum_P <- add_rownames(sum_Precision, "Class Name")
sum_Recall <- data.frame(LDA_R, QDA_R, KNN_R)
sum_R <- add_rownames(sum_Recall, "Class Name")

# ------------------------------------ Accuracy + Precision + Recall Comparison Graphs ------------------------------------
# Accuracy
AccuracyGraph <- ggplot(sum_model_Accuracy, aes(x=reorder(Model, -Accuracy), Accuracy)) + geom_point(stat="identity", aes(shape = Model, color = Model), size=4) + geom_text(aes(label = Model), hjust = 1.3, srt=0, size = 5, fontface= "bold") + theme(axis.text.x=element_blank())
mashAccuracy <- AccuracyGraph + labs(title = "Accuracy from 3 Models", x = "Model", y = "Accuracy Percentage") 

# Precision
PrecisionG_LDA <- ggplot(sum_P, aes(x=reorder(Model, -(sum_P$LDA_P)), sum_P$LDA_P), color=sum_P$LDA_P) + geom_bar(stat="identity", aes(fill = sum_P$LDA_P)) + geom_text(aes(label = sum_P$`Class Name`), hjust = 1.05, srt=90, size = 3, color = "white", fontface= "bold") + theme(axis.text.x=element_blank())
mashPrecisionG_LDA <- PrecisionG_LDA + labs(title = "Precision - LDA", x = "LDA", y = "Precision Percentage", fill="Precision") + scale_fill_gradient(low="seagreen4",high="darkorange")

PrecisionG_QDA <- ggplot(sum_P, aes(x=reorder(Model, -(sum_P$QDA_P)), sum_P$QDA_P), color=sum_P$QDA_P) + geom_bar(stat="identity", aes(fill = sum_P$QDA_P)) + geom_text(aes(label = sum_P$`Class Name`), hjust = 1.05, srt=90, size = 3, color = "white", fontface= "bold") + theme(axis.text.x=element_blank())
mashPrecisionG_QDA <- PrecisionG_QDA + labs(title = "Precision - QDA", x = "QDA", y = "Precision Percentage", fill="Precision") + scale_fill_gradient(low="seagreen4",high="darkorange")

PrecisionG_KNN <- ggplot(sum_P, aes(x=reorder(Model, -(sum_P$KNN_P)), sum_P$KNN_P), color=sum_P$KNN_P) + geom_bar(stat="identity", aes(fill = sum_P$KNN_P)) + geom_text(aes(label = sum_P$`Class Name`), hjust = 1.05, srt=90, size = 3, color = "white", fontface= "bold") + theme(axis.text.x=element_blank())
mashPrecisionG_KNN <- PrecisionG_KNN + labs(title = "Precision - KNN", x = "KNN", y = "Precision Percentage", fill="Precision") + scale_fill_gradient(low="seagreen4",high="darkorange")
Precidion_Grid <- grid.arrange(mashPrecisionG_LDA, mashPrecisionG_QDA, mashPrecisionG_KNN , ncol = 3)


# Recall
RecallG_LDA <- ggplot(sum_R, aes(x=reorder(Model, -(sum_R$LDA_R)), sum_R$LDA_R), color=sum_R$LDA_R) + geom_bar(stat="identity", aes(fill = sum_R$LDA_R)) + geom_text(aes(label = sum_R$`Class Name`), hjust = 1.05, srt=90, size = 3, color = "white", fontface= "bold") + theme(axis.text.x=element_blank())
mashRecallG_LDA <- RecallG_LDA + labs(title = "Recall - LDA", x = "LDA", y = "Recall Percentage", fill="Recall") + scale_fill_gradient(low="dimgray",high="darkgoldenrod1")

RecallG_QDA <- ggplot(sum_R, aes(x=reorder(Model, -(sum_R$QDA_R)), sum_R$QDA_R), color=sum_R$QDA_R) + geom_bar(stat="identity", aes(fill = sum_R$QDA_R)) + geom_text(aes(label = sum_R$`Class Name`), hjust = 1.05, srt=90, size = 3, color = "white", fontface= "bold") + theme(axis.text.x=element_blank())
mashRecallG_QDA <- RecallG_QDA + labs(title = "Recall - QDA", x = "QDA", y = "Recall Percentage", fill="Recall") + scale_fill_gradient(low="dimgray",high="darkgoldenrod1")

RecallG_KNN <- ggplot(sum_R, aes(x=reorder(Model, -(sum_R$KNN_R)), sum_R$KNN_R), color=sum_R$KNN_R) + geom_bar(stat="identity", aes(fill = sum_R$KNN_R)) + geom_text(aes(label = sum_R$`Class Name`), hjust = 1.05, srt=90, size = 3, color = "white", fontface= "bold") + theme(axis.text.x=element_blank())
mashRecallG_KNN <- RecallG_KNN + labs(title = "Recall - KNN", x = "KNN", y = "Recall Percentage", fill="Recall") + scale_fill_gradient(low="dimgray",high="darkgoldenrod1")
Recall_Grid <- grid.arrange(mashRecallG_LDA, mashRecallG_QDA, mashRecallG_KNN , ncol = 3)

# Put three graphs in one
grid.arrange(mashAccuracy, Precidion_Grid, Recall_Grid, nrow = 3)