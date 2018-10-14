#NLP airline tweets
#

rm(list = ls()) 

install.packages("install.load")
library(install.load)

listofPackages= c("DMwR","ROSE","wordcloud", "kernlab","caret","tm","dplyr","splitstackshape","e1071","SnowballC","randomForest","Metrics","gbm","wordcloud","RWeka")
install_load(listofPackages)


#Set working dir
workingDirectory = "C:/repo/NLPTweetAirlines/"
codeDirectory = paste(workingDirectory, "Code/", sep = "")
dataDirectory = paste(workingDirectory, "Data/", sep = "")
setwd(dataDirectory)
dataCSV<-read.csv('AirlineSentiment.csv')


#Read tweet data and perfom preproc
dataAll<-dataCSV[,colnames(dataCSV) %in% c("airline_sentiment","negativereason", "airline", "text")]
dataAll$text<-as.character(dataAll$text)
dataAll$negativereason <- as.character(dataAll$negativereason)
dataAll[dataAll$negativereason=="","negativereason"]<-"none Negative"


#Descriptive analytics
senPercentage<-prop.table(table(dataAll[,c("airline_sentiment","airline")]))*100
barplot(senPercentage, main="Airline sentiment tweets (% of neg, neutral, and pos)", 
        xlab="Airline",las=1)

tweetPercentage<-prop.table(table(dataAll[,c("airline_sentiment")]))*100
barplot(tweetPercentage, main="Sentiment (%) accross all airlines", 
        xlab="Sentiments",las=1)



negativeTweets<-dataAll[!(dataAll$negativereason=="none Negative"),]
negReason<-prop.table(table(negativeTweets[,c("negativereason")]))*100
ordlabel <- order(-negReason)
barplot(negReason[ordlabel], main="Percentage of each category of negative tweets", 
        xlab="Percentage",las=2,cex.names=0.5,horiz = TRUE)

negReason<-prop.table(table(negativeTweets[,c("negativereason")]))*100
ordlabel <- order(negReason)
barplot(negReason[ordlabel], main="Categories of negative tweets", 
        xlab="Percentage",las=2,cex.names=0.5,horiz = TRUE)


negativeTweetsUnited<-negativeTweets[negativeTweets$airline=="United",]
negReason<-prop.table(table(negativeTweetsUnited[,c("negativereason")]))*100
ordlabel <- order(negReason)
barplot(negReason[ordlabel], main="Categories of negative tweets for United", 
        xlab="Percentage",las=2,cex.names=0.5,horiz = TRUE)

negativeTweetsUnited<-negativeTweets[negativeTweets$airline=="Virgin America",]
negReason<-prop.table(table(negativeTweetsUnited[,c("negativereason")]))*100
ordlabel <- order(negReason)
barplot(negReason[ordlabel], main="Categories of negative tweets for Virgin America", 
        xlab="Percentage",las=2,cex.names=0.5,horiz = TRUE)


#Predictive analytics
reasonorSetiment<-"negativereason"
reason = "negativereason"
#reason = "sentiment"

sentiment = "airline_sentiment"

outputOfInterest<-reason


if(reasonorSetiment == reason)
{
  myData<-dataAll[dataAll$airline_sentiment=="negative",!(colnames(dataAll) %in% sentiment)]  
}else
{
  myData<-dataAll[,!(colnames(dataAll) %in% reason)]
}

tweetTxt<-myData$text

#corpus preproc
corpus <- VCorpus(VectorSource(tweetTxt))
corpus <- tm_map(corpus, content_transformer(stripWhitespace))
corpus <- tm_map(corpus, content_transformer(tolower))
corpus <- tm_map(corpus, content_transformer(removeNumbers))
corpus <- tm_map(corpus, content_transformer(removePunctuation))
corpus <- tm_map(corpus, stemDocument, language = "english") 
corpus <- tm_map(corpus, content_transformer(gsub), pattern = "flightl|flighted|fligh|flighted|flightr|flightled|flightalation|flight", replacement = "flight")
corpus <- tm_map(corpus, content_transformer(gsub), pattern = "X.ûïjetblu", replacement = "jetblu")
stopwords <- c(stopwords("english"), "usairway", "southwestair", "americanair", "jetblu","delta", "united", "virginamerica")
corpus <- tm_map(corpus, removeWords, stopwords) #remove stop words


#create document term freq matrix
biTokenizer <- function(x) NGramTokenizer(x, Weka_control(min=1, max=1))
tdm <- DocumentTermMatrix(corpus, control=list(tokenize = biTokenizer))
inspect(tdm)

#create term freq
freq<-(colSums(as.matrix(tdm)))

#check most frequent terms for replacing terms
ord <- order(freq,decreasing=TRUE)
freq[head(ord,200)]


set.seed(42)
#limit words by specifying min frequency
wordcloud(names(freq),freq, min.freq=70)



hist(colSums(as.matrix(tdm)), 
     main="Histogram of word freq", 
     xlab="freq", 
     border="blue", 
     col="green",
     xlim=c(1,1000),
     ylim=c(1,10000),
     las=1, 
     breaks=100)

#remove 99% sparse features (this is good for sentiment)
#dtm <- removeSparseTerms(tdm, .99)
feature_list <- findFreqTerms(tdm, lowfreq=5, highfreq =  300)



# Convert to a data.frame for training and assign a classification (factor) to each document.
tweet.df <- data.frame(as.matrix(tdm[,feature_list]), stringsAsFactors=FALSE)




tweet.df <- cbind(tweet.df,as.factor(myData[,outputOfInterest]))
colnames(tweet.df)[ncol(tweet.df)] <- outputOfInterest

if(reasonorSetiment == reason)
{
#creating word cloud
set.seed(12)
thisTypeofTweet<-tweet.df[tweet.df[,outputOfInterest] == "Customer Service Issue",!(colnames(tweet.df) %in% outputOfInterest)]
freq<-colSums(thisTypeofTweet)
#limit words by specifying min frequency
wordcloud(names(freq),freq, min.freq = 50)
}else
{
set.seed(12)
thisTypeofTweet<-tweet.df[tweet.df[,outputOfInterest] == "positive",!(colnames(tweet.df) %in% outputOfInterest)]
freq<-colSums(thisTypeofTweet)
#limit words by specifying min frequency
wordcloud(names(freq),freq, min.freq = 30)



set.seed(12)
thisTypeofTweet<-tweet.df[tweet.df[,outputOfInterest] == "neutral",!(colnames(tweet.df) %in% outputOfInterest)]
freq<-colSums(thisTypeofTweet)
#limit words by specifying min frequency
wordcloud(names(freq),freq, min.freq = 50)
}


#building ML
#creating train and test set
trainingPercentage <-0.80
set.seed(364) 
inx <- createDataPartition(y = tweet.df[,outputOfInterest], times = 1, p = trainingPercentage, list = FALSE)
trainDataSet<-tweet.df[inx,]
testDataSet<-tweet.df[-inx,]

#building ML
ntrees=50

counts <- table(trainDataSet[,outputOfInterest])
nRareSamples = min(counts)
nRareSamplesPercentage = 0.9
fmla <- formula(paste(outputOfInterest, "~.")) #formula object

#######################oversampling rf
#trControl = trainControl(method = "cv", number = 3, allowParallel = TRUE, verboseIter = FALSE, sampling = "rose")

#rf_model<-train(fmla,data=trainDataSet,method="rf",
#                ntree = 1,
#                trControl=trControl,
#                prox=TRUE,allowParallel=TRUE)

#testclass <- predict(rf_model, newdata = testDataSet)
#print(rf_model)


trControl = trainControl(method = "cv", number = 4, allowParallel = TRUE, verboseIter = FALSE, sampling = "smote")

rf_model<-train(fmla,data=trainDataSet,method="rf", ntree = ntrees,
                trControl=trControl,
                prox=TRUE,allowParallel=TRUE)

testclass <- predict(rf_model, newdata = testDataSet)
print(rf_model)

confusionMatrix(testclass,testDataSet[,outputOfInterest])

#######################balanced sample approach 
counts <- table(trainDataSet[,outputOfInterest])
nRareSamples = min(counts)
nRareSamplesPercentage = 0.9
fmla <- formula(paste(outputOfInterest, "~.")) #formula object

rftree.strata <- randomForest(fmla, data=trainDataSet,
                       strata=trainDataSet[,outputOfInterest],
                       sampsize=c(rep(floor(nRareSamples*nRareSamplesPercentage),nlevels(trainDataSet[,outputOfInterest]))) ,
                       importance=TRUE, 
                       ntree=ntrees,  
                       do.trace = TRUE)
print(rftree.strata)

#######################WT approach 
counts <- table(trainDataSet[,outputOfInterest])
sortedCounts<-sort(counts, decreasing  = TRUE)
sortedCounts1<-sortedCounts/sum(sortedCounts)
counter<-1
sortedCounts[counter]<-1
while(counter<nrow(counts))
{
  sortedCounts[counter+1]<-(sortedCounts1[counter]/sortedCounts1[counter+1])*sortedCounts[counter]
  counter<-counter+1
}
sortedCounts<-as.data.frame(sortedCounts)
sortedCounts <- sortedCounts[order(as.character(sortedCounts$Var1)),] 


classwtRF=sortedCounts$Freq
rftree.wt <- randomForest(fmla, data=trainDataSet,
                              classwt=classwtRF,
                              importance=TRUE, 
                              ntree=ntrees, 
                              do.trace = TRUE)
print(rftree.wt)

#######################CUToff approach 
counts <- table(trainDataSet[,outputOfInterest])
sortedCounts<-sort(counts, decreasing  = TRUE)
sortedCounts1<-sortedCounts/sum(sortedCounts)
counter<-1
sortedCounts[counter]<-1
while(counter<nrow(counts))
{
  sortedCounts[counter+1]<-(sortedCounts1[counter]/sortedCounts1[counter+1])*sortedCounts[counter]
  counter<-counter+1
}
sortedCounts<-as.data.frame(sortedCounts)
sortedCounts <- sortedCounts[order(as.character(sortedCounts$Var1)),] 


cutoffRF=sortedCounts$Freq/sum(classwtRF)
#reshape(as.data.frame(sortedCounts), timevar = "Var1", direction = "wide")
rftree.ct <- randomForest(fmla, data=trainDataSet,
                          cutoff=cutoffRF,
                          importance=TRUE, 
                          ntree=ntrees, 
                          do.trace = TRUE)
print(rftree.ct)

#######################GBM simple approach 
gbm.simple<-gbm(fmla, data=trainDataSet,n.trees=ntrees,shrinkage=0.01, cv.folds = 4, distribution="multinomial", verbose=TRUE)
best.iter.simple <- gbm.perf(gbm.simple,method="cv")
print(best.iter.simple)
f.predict <- predict(gbm.simple,testDataSet, best.iter.simple, type = "response")

#######################GBM weighted approach 
numberOfTraining<-nrow(trainDataSet)
model_weights<-c(1:numberOfTraining)
totalClasses<-nlevels(trainDataSet[,outputOfInterest])
while(counter<=totalClasses)
{
  model_weights[trainDataSet[,outputOfInterest] == unique(trainDataSet[,outputOfInterest])[counter]]<-((1/table(trainDataSet[,outputOfInterest])[counter]) * 1/totalClasses)
  counter<-counter+1
}
gbm.wt<-gbm(fmla, data=trainDataSet,n.trees=ntrees,shrinkage=0.01, cv.folds = 4, distribution="multinomial", verbose=TRUE, weights = model_weights)
best.iter.wt <- gbm.perf(gbm.wt,method="cv")
print(best.iter.wt)
f.predict <- predict(gbm.wt,testDataSet, best.iter.wt,type = "response")

#view OOB-CV specificity and sensitiviy
#plot(roc(rftree.ct$votes[,2],trainDataSet$negativereason),main="black default, red stata, green classwt")
#plot(roc(rftree.strata$votes[,2],trainDataSet$negativereason),col=2,add=T)
#plot(roc(rftree.wt$votes[,2],trainDataSet$negativereason),col=3,add=T)

## Evaluating the models
#Random forest prediction performance
result.rf<-sapply(c("rftree.strata","rftree.wt","rftree.ct"),function(a.model) {
  ce(testDataSet[,outputOfInterest] , predict(get(a.model), newdata=testDataSet))
})

#GBM prediction performance
result.gbm<-sapply(c("gbm.simple", "gbm.wt"),function(a.model) {
  ce(as.numeric(testDataSet[,outputOfInterest]) , apply(predict(get(a.model),testDataSet, ifelse(a.model=="gbm.simple",best.iter.simple,best.iter.wt), type = "response"), 1, which.max))
})

#confusion matrix for RF
caret::confusionMatrix(data = predict(get("rftree.wt"), newdata=testDataSet), reference = testDataSet[,outputOfInterest], positive="1", mode="everything")
#confusion matrix for GBM
confusionMatrix(data = as.factor(as.numeric(apply(predict(get("gbm.wt"),testDataSet, type = "response"), 1, which.max))), reference = as.factor(as.numeric(testDataSet[,outputOfInterest])) )


save(file="workPlace.negReason.RData")



