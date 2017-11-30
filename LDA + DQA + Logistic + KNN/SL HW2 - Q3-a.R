# Q3 - a : NLP for Spam or Ham then build a Niave Bayes Model.
# Reference - https://rpubs.com/cen0te/naivebayes-sentimentpolarity

# Library
library(caret)
library(tm)          # Word Processing
library(wordcloud)   # Word Cloud
library(RColorBrewer)# Color Code
library(e1071)       # Naive Bayes
library(MLmetrics)   # Calculate Correlation Coeffcient
library(stringr)     # string replace
library(stringi)     # Convert string to UTF-8
library(cooccur)
library(textmining)  # Brew Word Cloud
library(ggplot2)     # Graphs
library(gridExtra)   # Combined Graphs into One

# --------------------------------------------------- Pre-Procssed Data ---------------------------------------------------
# Load in data as qutoed string
SMSS_raw <- readLines("SMSSpamCollection.txt")

# Replace \t with ,
untab_SMSS <- str_replace_all(SMSS_raw, "\t", ",")

# Replace double quote by single quote in the free text
doubleQuote_SMSS <- str_replace_all(untab_SMSS, "\"", "'")

# lowercase all strings
#lower_SMSS <- tolower(doubleQuote_SMSS)

# Extract "Type" Column
type_Col_S <- substr(doubleQuote_SMSS, 1, 4)
type <- str_replace_all(type_Col_S, ",", "")

# Extract "Text" Column
text_Col_S <- substr(doubleQuote_SMSS, 5, nchar(doubleQuote_SMSS))
text <- ifelse(substring(text_Col_S, 1, 1) == ",",str_replace(text_Col_S, ",", ""), text_Col_S)

# Covert text to utf-8 format
text <- stri_enc_toutf8(text,  is_unknown_8bit = FALSE, validate = FALSE)

# Dataframe for the data
SMSS_DF <- data.frame(type, text)

# Type as factor
SMSS_DF$type <- factor(SMSS_DF$type)


# Parse data randomly into 75% Training and 25% Testing
all <- (1:nrow(SMSS_DF))
TrainRandom <- sample(1:nrow(SMSS_DF), as.integer(nrow(SMSS_DF)*0.75), replace=F) # Take 4180 data randomly as Training
TestRandom <- setdiff(all, TrainRandom)     # Rest of the data as Testing


# Train and Test Dataset
TrainSet <- SMSS_DF[TrainRandom,]
TestSet <- SMSS_DF[TestRandom,]

# Ham messages
trainData_ham <- TrainSet[TrainSet$type == "ham",]
head(trainData_ham$text)
tail(trainData_ham$text)

#spam messages
trainData_spam <- TrainSet[TrainSet$type == "spam",]
head(trainData_spam$text)
tail(trainData_spam$text)

# ---------------------------------------------------- Comb Through Data (Training) ----------------------------------------------------
#create the corpus
corpus <- Corpus(VectorSource(TrainSet$text))

# Normalize to lowercase 
corpus <- tm_map(corpus, content_transformer(tolower))

# Remove numbers
corpus <- tm_map(corpus, removeNumbers)

# Remove stopwords
corpus <- tm_map(corpus, removeWords, stopwords("english"))

# Remove punctuation
corpus <- tm_map(corpus, removePunctuation)

# Normalize whitespaces
corpus <- tm_map(corpus, stripWhitespace)

# -------------------------------------------------------- Brew Word Cloud -----------------------------------------------------------
# Ham Word Cloud
termDocMatrix_Ham <- TermDocumentMatrix(corpus[TrainSet$type == "ham"])
termDocMatrix_MHam <- as.matrix(termDocMatrix_Ham)
Ham_sort_TDM <- sort(rowSums(termDocMatrix_MHam), decreasing=TRUE)
Ham_term_DF <- data.frame(word = names(Ham_sort_TDM), freq = Ham_sort_TDM)
head(Ham_term_DF, 10)
Ham_wordCloud <- wordcloud(words = Ham_term_DF$word, freq = Ham_term_DF$freq, min.freq = 10, max.words = 300, random.order = FALSE, rot.per = 0.35, colors = rev(brewer.pal(8, "RdYlGn")))

# Spam Word Cloud
termDocMatrix_Spam <- TermDocumentMatrix(corpus[TrainSet$type == "spam"])
termDocMatrix_MSpam <- as.matrix(termDocMatrix_Spam)
Spam_sort_TDM <- sort(rowSums(termDocMatrix_MSpam), decreasing=TRUE)
Spam_term_DF <- data.frame(word = names(Spam_sort_TDM), freq = Spam_sort_TDM)
head(Spam_term_DF, 10)

Spam_wordCloud <- wordcloud(words = Spam_term_DF$word, freq = Spam_term_DF$freq, min.freq = 1, max.words = 300, random.order = FALSE, rot.per = 0.35, colors = rev(brewer.pal(8, "RdGy")))

# Top 10 words for Spam and Ham
SpamFrqGraph <- ggplot(Spam_term_DF[1:10,], aes(x=reorder(Spam_term_DF$word[1:10], -(Spam_term_DF$freq[1:10])), Spam_term_DF$freq[1:10]), color=Spam_term_DF$freq[1:15]) + geom_bar(stat="identity", aes(fill = Spam_term_DF$freq[1:10])) + geom_text(aes(label = Spam_term_DF$word[1:10]), hjust = 1.05, srt=90, size = 4, color = "white", fontface= "bold") + theme(axis.text.x=element_blank())
mashSpam <- SpamFrqGraph + labs(title = "Top 10 Words for Spam", x = "Top 10 Words", y = "Frequency Counts", fill = "Frequency Counts") + scale_fill_gradient(low="skyblue2",high="tomato")

HamFrqGraph <- ggplot(Ham_term_DF[1:10,], aes(x=reorder(Ham_term_DF$word[1:10], -(Ham_term_DF$freq[1:10])), Ham_term_DF$freq[1:10]), color=Ham_term_DF$freq[1:15]) + geom_bar(stat="identity", aes(fill = Ham_term_DF$freq[1:10])) + geom_text(aes(label = Ham_term_DF$word[1:10]), hjust = 1.05, srt=90, size = 4, color = "white", fontface= "bold") + theme(axis.text.x=element_blank())
mashHam <- HamFrqGraph + labs(title = "Top 10 Words for Ham", x = "Top 10 Words", y = "Frequency Counts", fill = "Frequency Counts") + scale_fill_gradient(low="seagreen4",high="darkorange")

# Combine two graphs into one
grid.arrange(mashSpam, mashHam, ncol = 2)

# --------------------------------------------------- Naive Bayes -------------------------------------------------
# Tokenization for all the data

# Create DTM with at least 2 characters
all_DTM <- DocumentTermMatrix(corpus, control = list(global = c(2, Inf)))

# Observe the content term matrix
T_Matrix <- inspect(all_DTM[1:30, 1:13])

# Pick out words that appears at least 5 times
all_features <- findFreqTerms(all_DTM, 5)

# Peek into part of the results
summary(all_features)
head(all_features)

# Train and test the words that exists in the dictionary
word_DTM_Train <- DocumentTermMatrix(corpus, list(global = c(2, Inf), dictionary = all_features, repeats=100))

# Create categorical features ( Labeled )
convert_cat_count <- function(x){
  x <- ifelse(x > 0, 1, 0)
  x <- factor(x, levels = c(0, 1), labels = c("F", "T"))
  return (x)
}
# Add labels to the DTM Train Data
word_DTM_Train <- apply(word_DTM_Train, MARGIN = 2, convert_cat_count)

# Train Naive Bayes Model
SMS_NB <- naiveBayes(word_DTM_Train, TrainSet$type)

# Peek the result
SMS_NB[[2]][1:5]

# ---------------------------------------------------- Comb Through Data (Testing) ----------------------------------------------------
#create the corpus
corpus_T <- Corpus(VectorSource(TestSet$text))

# Normalize to lowercase 
corpus_T <- tm_map(corpus_T, content_transformer(tolower))

# Remove numbers
corpus_T <- tm_map(corpus_T, removeNumbers)

# Remove stopwords
corpus_T <- tm_map(corpus_T, removeWords, stopwords("english"))

# Remove punctuation
corpus_T <- tm_map(corpus_T, removePunctuation)

# Normalize whitespaces
corpus_T <- tm_map(corpus_T, stripWhitespace)

# Test Data
word_DTM_Test <- DocumentTermMatrix(corpus_T, list(global = c(2, Inf), dictionar = all_features))
word_DTM_Test <- apply(word_DTM_Test, MARGIN = 2, convert_cat_count)

# Evaluate Naive Bayes Model
NB_Predict_M <- predict(SMS_NB, word_DTM_Test)

# Confusion Matrix summary in a table
ConfusionM_Table <- table(TestSet$type, NB_Predict_M)

# Confusion Matrix
ConfusionM_NB <- ConfusionMatrix(NB_Predict_M, TestSet$type)

# Accuracy, Precision and Recall for Naive Bayes
NB_Accuracy <- Accuracy(NB_Predict_M, TestSet$type)
NB_Precision <- Precision(NB_Predict_M, TestSet$type)
NB_Recall <- Recall(NB_Predict_M, TestSet$type)