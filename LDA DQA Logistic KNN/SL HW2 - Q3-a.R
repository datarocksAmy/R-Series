# Q3 : MLP for Spam or Ham

# Library
library(caret)
library(tm)
library(wordcloud)
library(e1071)
library(MLmetrics)
library(stringr) # string replace
library(stringi) # Convert string to UTF-8
library(cooccur)

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

# Set step size 
set.seed(400)

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

# ---------------------------------------------------- Comb Through Data ----------------------------------------------------
#create the corpus
corpus <- Corpus(VectorSource(TrainSet$text))
# Normalize to lowercase (not a standard tm transformation)
corpus <- tm_map(corpus, content_transformer(tolower))
# Remove numbers
corpus <- tm_map(corpus, removeNumbers)
# Remove stopwords e.g. to, and, but, or (using predefined set of word in tm package)
corpus <- tm_map(corpus, removeWords, stopwords())
# Remove punctuation
corpus <- tm_map(corpus, removePunctuation)
# Normalize whitespaces
corpus <- tm_map(corpus, stripWhitespace)

# -------------------------------------------------------- Brew Word Cloud -----------------------------------------------------------
pal1 <- brewer.pal(9,"YlGn")
pal1 <- pal1[-(1:4)]

pal2 <- brewer.pal(9,"Reds")
pal2 <- pal2[-(1:4)]


par(mfrow = c(1,2))
wordcloud(corpus[TrainSet$type == "ham"], min.freq = 30, random.order = FALSE, colors = pal1)
wordcloud(corpus[TrainSet$type == "spam"], min.freq = 30, random.order = FALSE, colors = pal2)