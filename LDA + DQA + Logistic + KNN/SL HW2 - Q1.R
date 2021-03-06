# Q1-a-i : Using only Events for "Snow" and "Rain" from "kc_weather" to build models for LDA, QDA and KNN. 
#          Split the dataset into training and testing for over 100+ iterations.

# Import Library
library(lubridate) # Extract Date to Year, Month, Day
library(caret)  # Logitic Regression
library(MASS)   # LDA
library(e1071)
library(class)  # KNN
library(ggplot2)
library(gridExtra)
library(LiblineaR)

# Read in dataset kc_weather
KCWeather = read.csv("kc_weather.csv", header=TRUE)
head(KCWeather)

# Seperate Date into Year, Month, Day
KCWeather$Year <- year(KCWeather$Date)
KCWeather$Month <- month(KCWeather$Date)
KCWeather$Day <- day(KCWeather$Date)

# Subset for Events Rain and Snow from KCWeather dataset
newKCW <- subset(KCWeather, Events != "Rain_Thunderstorm")
#levels(newKCW$Events)[levels(newKCW$Events)=="Snow"] <- 0
#levels(newKCW$Events)[levels(newKCW$Events)=="Rain"] <- 1

# Change level for subset : Snow and Rain
newKCW$Events <- factor(newKCW$Events)

# Numerical Labels for Events : Snow = 1, Rain = 0
newKCW$EventLabel <- ifelse(newKCW$Events == "Rain", 0 ,1)

# Parse data randomly into 80% Training and 20% Testing
all <- (1:226)
TrainRandom<- sample(1:226, 181, replace=F)
TestRandom <- setdiff(all, TrainRandom)

# Train and Test Dataset
TrainSet <- newKCW[TrainRandom,]
TestSet <- newKCW[TestRandom,]

# Labels for Training and Testing 
TrainEvents <- newKCW$Events[TrainRandom]
TestEvents <- newKCW$Events[TestRandom]

# ----------------------------------------- LOGISTIC REGRESSION -----------------------------------------
# Build Logitstic Regression Model & Train with train dataset
# LogiReg_Model <- glm(Events ~ Temp.F + Dew_Point.F + Humidity.percentage + Sea_Level_Press.in + Visibility.mi + Wind.mph + Precip.in + Year + Month + Day, data = TrainSet, family=binomial("logit"))
LogiReg_Model <- train(Events ~ Temp.F + Dew_Point.F + Humidity.percentage + Sea_Level_Press.in + Visibility.mi + Wind.mph + Precip.in + Year + Month + Day, data = TrainSet, method = 'regLogistic', repeats=100)
summary(LogiReg_Model)

# Test Model with test dataset
ProbLR <- predict(LogiReg_Model, TestSet, type = "prob")

# Anova Test
#LR_anova <- anova(LogiReg_Model, test="Chisq")

# Predict with Test Data
predict_weather = rep("Rain", nrow(TestSet))
predict_weather[ProbLR$Snow > .5] = "Snow"   

# Confusion Matrix
LogisticRegr_Confusion <- confusionMatrix(data = predict_weather, reference = TestEvents, mode = "prec_recall")

# Accuracy + Precision + Recall
LR_Accuracy <- LogisticRegr_Confusion$overall["Accuracy"]
LR_Precision <- LogisticRegr_Confusion$byClass["Precision"]
LR_Recall <- LogisticRegr_Confusion$byClass["Recall"]

# --------------------------------------------------------- LDA ---------------------------------------------------------
# Train LDA Model
#LDA_Model <- lda(Events ~ Temp.F + Dew_Point.F + Humidity.percentage + Sea_Level_Press.in + Visibility.mi + Wind.mph + Precip.in + Year + Month + Day, data = TrainSet)
LDA_Model <- train(Events ~ Temp.F + Dew_Point.F + Humidity.percentage + Sea_Level_Press.in + Visibility.mi + Wind.mph + Precip.in + Year + Month + Day, data = TrainSet, method = 'lda', repeats=100)

# Test LDA Model
LDA_Predict <- predict(LDA_Model, TestSet)

# Confusion Matrix
#LDA_Class <- LDA_Predict$class  # when run on lda() command
table(LDA_Predict, TestEvents)

# Test Error Rate 
mean(LDA_Predict == TestEvents)

# Confusion Matrix ( Accuracy, Precision, Recall )
LDA_Confusion <- confusionMatrix(data = LDA_Predict, reference = TestEvents, mode = "prec_recall")

# Accuracy + Precision + Recall
LDA_Accuracy <- LDA_Confusion$overall["Accuracy"]
LDA_Precision <- LDA_Confusion$byClass["Precision"]
LDA_Recall <- LDA_Confusion$byClass["Recall"]

# --------------------------------------------------------- QDA ---------------------------------------------------------
# Train QDA Model
#QDA_Model <- qda(Events ~ Temp.F + Dew_Point.F + Humidity.percentage + Sea_Level_Press.in + Visibility.mi + Wind.mph + Precip.in + Year + Month + Day, data = TrainSet)
QDA_Model <- train(Events ~ Temp.F + Dew_Point.F + Humidity.percentage + Sea_Level_Press.in + Visibility.mi + Wind.mph + Precip.in + Year + Month + Day, data = TrainSet, method = 'qda', repeats = 100)

# Test QDA Model
QDA_Predict <- predict(QDA_Model, TestSet)

# Confusion Matrix
#QDA_Class <- QDA_Predict$class  # When run on qda() command
table(QDA_Predict, TestEvents)

# Test Error Rate 
mean(QDA_Predict == TestEvents)

# Confusion Matrix ( Accuracy, Precision, Recall )
QDA_Confusion <- confusionMatrix(data = QDA_Predict, reference = TestEvents, mode = "prec_recall")

# Accuracy + Precision + Recall
QDA_Accuracy <- QDA_Confusion$overall["Accuracy"]
QDA_Precision <- QDA_Confusion$byClass["Precision"]
QDA_Recall <- QDA_Confusion$byClass["Recall"]

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
KNN_Precision <- KNN_Confusion$byClass["Precision"]
KNN_Recall <- KNN_Confusion$byClass["Recall"]

# --------------------------------------------------------- COMPARE MODELS ----------------------------------------------------------
Model <- c("Logistic","LDA", "QDA", "KNN")
Accuracy <- c(LR_Accuracy ,LDA_Accuracy, QDA_Accuracy, KNN_Accuracy)
Precision <- c(LR_Precision, LDA_Precision, QDA_Precision, KNN_Precision)
Recall <- c(LR_Recall ,LDA_Recall, QDA_Recall, KNN_Recall)
summary_df <- data.frame(Model, Accuracy, Precision, Recall)

# ------------------------------------ Accuracy + Precision + Recall Comparison Graphs ------------------------------------
# Accuracy
AccuracyGraph <- ggplot(summary_df, aes(x=reorder(Model, -Accuracy), Accuracy), color=Accuracy) + geom_bar(stat="identity", aes(fill = Accuracy)) + geom_text(aes(label = Model), hjust = 1.3, srt=90, size = 3.5, color = "white", fontface= "bold") + theme(axis.text.x=element_blank())
mashAccuracy <- AccuracyGraph + labs(title = "Accuracy from 4 Models", x = "Model") + scale_fill_gradient(low="skyblue2",high="tomato")
# Precision
PrecisionGraph <- ggplot(summary_df, aes(x=reorder(Model, -Precision), Precision), color=Precision) + geom_bar(stat="identity", aes(fill = Precision)) + geom_text(aes(label = Model), hjust = 1.3, srt=90, size = 3.5, color = "white", fontface= "bold") + theme(axis.text.x=element_blank())
mashPrecision <- PrecisionGraph + labs(title = "Precision from 4 Models", x = "Model") + scale_fill_gradient(low="seagreen4",high="darkorange")
# Recall
RecallGraph <- ggplot(summary_df, aes(x=reorder(Model, -Recall), Recall), color=Precision) + geom_bar(stat="identity", aes(fill = Recall)) + geom_text(aes(label = Model), hjust = 1.3, srt=90, size = 3.5, color = "white", fontface= "bold") + theme(axis.text.x=element_blank())
mashRecall <- RecallGraph + labs(title = "Recall from 4 Models", x = "Model") + scale_fill_gradient(low="dimgray",high="darkgoldenrod1")

# Put three graphs in one
grid.arrange(mashAccuracy, mashPrecision, mashRecall, ncol = 3)

