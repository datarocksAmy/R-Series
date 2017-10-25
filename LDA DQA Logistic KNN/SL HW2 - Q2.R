# Q2 : Naive Bayes for Qualitative Predictors 

# Import Library
library(class)      # KNN
library(lubridate)  # Extract Date to Year, Month, Day
library(e1071)      # Naive Bayes
library(caret)      # Confusion Matrix
library(MLmetrics)  # Calculate Correlation Coeffcient


# Read in dataset kc_weather
KCWeather = read.csv("kc_weather.csv", header=TRUE)
head(KCWeather)

# Numerical Labels for Events
KCWeather$EventLabel <- ifelse(KCWeather$Events == "Rain",0 , ifelse(KCWeather$Events == "Snow", 1, 2))

# Seperate Date into Year, Month, Day
KCWeather$Year <- year(KCWeather$Date)
KCWeather$Month <- month(KCWeather$Date)
KCWeather$Day <- day(KCWeather$Date)

# Observe data summary for Temp.F, Humidity.percentage, Precip.in
summary(KCWeather$Temp.F)
summary(KCWeather$Humidity.percentage)
summary(KCWeather$Precip.in)

# Qualitative categories for predictors Temp.F, Humidity.percentage, Precip.in
KCWeather$TempF_Q <- ifelse(KCWeather$Temp.F <= 46,"T1" , ifelse(KCWeather$Temp.F > 46 & KCWeather$Temp.F <= 62.5, "Tm", "T3"))
KCWeather$Humidity_Q <- ifelse(KCWeather$Humidity.percentage <= 62,"H1" , ifelse(KCWeather$Humidity.percentage > 62 & KCWeather$Humidity.percentage <= 72, "Hm", "H3"))
KCWeather$Precip_Q <- ifelse(KCWeather$Precip.in <= 0.025,"Pm" , ifelse(KCWeather$Precip.in > 0.025 & KCWeather$Precip.in <=0.1728, "Pme", "P3"))


# Parse data randomly into Training 290 and the rest of data for Testing
all <- (1:366)
TrainRandom<- sample(1:366, 290, replace=F)
TestRandom <- setdiff(all, TrainRandom)

# Train and Test Dataset
TrainSet <- KCWeather[TrainRandom,]
TestSet <- KCWeather[TestRandom,]

# Labels for Training and Testing 
TrainEvents <- KCWeather$Events[TrainRandom]
TestEvents <- KCWeather$Events[TestRandom]
TestLabels <- data.frame(TestSet$TempF_Q, TestSet$Humidity_Q, TestSet$Precip_Q)

# Convert to factors
KCWeather$TempF_Q <- as.factor(KCWeather$TempF_Q)
KCWeather$Humidity_Q <- as.factor(KCWeather$Humidity_Q)
KCWeather$Precip_Q <- as.factor(KCWeather$Precip_Q)

# Naive Bayes
#NaiveBayesModel <- naiveBayes(Events ~ TempF_Q + Humidity_Q + Precip_Q, data = TrainSet)
NaiveBayesModel  <- train(Events ~ TempF_Q + Humidity_Q + Precip_Q, method = "naive_bayes", repeats = 100, data = TrainSet)

# Predict using Test Dat
#testing <- data.frame(TempF_Q = c("T1"), Humidity_Q=c("H1"), Precip_Q=c("P3"))
NB_Predict <- predict(NaiveBayesModel, TestSet, type="raw")
table(NB_Predict, TestEvents)

# Confusion Matrix
NB_Confusion <- confusionMatrix(NB_Predict, TestEvents,  mode = "prec_recall")

# Accuracy + Precision + Recall
NB_Accuracy <- Accuracy(NB_Predict, TestEvents)
NB_Precision <- data.frame(NB_Confusion$byClass[, "Precision"])
NB_Recall <- data.frame(NB_Confusion$byClass[,"Recall"])

# Summary Precision/Recall
sum_P_NB <- tibble::rownames_to_column(NB_Precision, "Class Name")
sum_R_NB <- tibble::rownames_to_column(NB_Recall, "Class Name")

# ------------------------------------ Precision + Recall Comparison Graphs ------------------------------------
# Accuracy
#AccuracyGraph <- ggplot(summary_df, aes(x=reorder(Model, -NB_Accuracy), Accuracy), color=Accuracy) + geom_bar(stat="identity", aes(fill = Accuracy)) + geom_text(aes(label = Model), hjust = 1.3, srt=90, size = 3.5, color = "white", fontface= "bold") + theme(axis.text.x=element_blank())
#mashAccuracy <- AccuracyGraph + labs(title = "Accuracy", x = "Model") + scale_fill_gradient(low="skyblue2",high="tomato")

# Precision
Precision_NB <- ggplot(sum_P_NB, aes(x=reorder(sum_P_NB$`Class Name`, -(sum_P_NB$NB_Confusion.byClass....Precision..)), sum_P_NB$NB_Confusion.byClass....Precision..), color=sum_P_NB$NB_Confusion.byClass....Precision..) + geom_bar(stat="identity", aes(fill = sum_P_NB$NB_Confusion.byClass....Precision..)) + geom_text(aes(label = sum_P_NB$`Class Name`), hjust = 1.05, srt=90, size = 4, color = "white", fontface= "bold") + theme(axis.text.x=element_blank())
mashPrecision_NB <- Precision_NB + labs(title = "Precision - Naive Bayes", x = "Classes", y = "Precision Percentage", fill="Precision") + scale_fill_gradient(low="seagreen4",high="darkorange")


# Recall
Recall_NB <- ggplot(sum_R_NB, aes(x=reorder(sum_R_NB$`Class Name`, -(sum_R_NB$NB_Confusion.byClass....Recall..)), sum_R_NB$NB_Confusion.byClass....Recall..), color=sum_R_NB$NB_Confusion.byClass....Recall..) + geom_bar(stat="identity", aes(fill = sum_R_NB$NB_Confusion.byClass....Recall..)) + geom_text(aes(label = sum_P_NB$`Class Name`), hjust = 1.05, srt=90, size = 4, color = "white", fontface= "bold") + theme(axis.text.x=element_blank())
mashRecall_NB <- Recall_NB + labs(title = "Recall - Naive Bayes", x = "Classes", y = "Recall Percentage", fill="Recall") + scale_fill_gradient(low="dimgray",high="darkgoldenrod1")

# Put three graphs in one
grid.arrange(mashPrecision_NB, mashRecall_NB, ncol = 2)