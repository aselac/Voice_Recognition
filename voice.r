#Loading libraries
if (!require(tidyverse)) install.packages('tidyverse')
if (!require(caret)) install.packages('caret')
if (!require(rpart)) install.packages('rpart')
if (!require(knitr)) install.packages('knitr')
if (!require(knitr)) install.packages('matrixStats')
library(tidyverse)
library(caret)
library(rpart)
library(knitr)
library(matrixStats)


#https://www.kaggle.com/primaryobjects/voicegender 
data<-read.csv("https://raw.githubusercontent.com/aselac/Voice_Recognition/master/voice.csv")

colnames(data)[colnames(data) == "label"] <- "gender" # change column names
str(data) # inspect the file
sum(is.na(data)) # check missing values

# Find count of males and females
data%>%group_by(gender)%>%summarise(count=n())%>%kable

# variation of mean frequency across genders
data%>%ggplot(aes(gender,meanfreq, fill=gender))+geom_boxplot()+ggtitle("Mean frequency variation")

# predictor variation across labels
data%>%gather(key,value,1:20)%>%ggplot(aes(value,fill=gender))+geom_density(alpha=0.3)+
  facet_wrap(~key,scales = "free")+ggtitle("Variation of Predictors Across Gender")


# Heat map to discover clusters
x_centered <- sweep(data[-21], 2, colMeans(data[-21])) #reduce colMeans
x_scaled <- sweep(x_centered, 2, colSds(as.matrix(data[-21])), FUN = "/") # divide by Sd 
d_features <- dist(t(x_scaled)) 
heatmap(as.matrix(d_features), labRow = NA, labCol = NA)

# find values which correlates with each other
cor_data<-cor(data[-21])

# remove values over 0.7 which has high correlated (need to remove predictive bias)
highly_cor<-colnames(data)[findCorrelation(cor_data, cutoff = 0.7)]
highly_cor
data_clean<-data[, which(!colnames(data) %in% highly_cor)]



#data splitting
set.seed(1, sample.kind = "Rounding")
test_index<-createDataPartition(data_clean$gender,  times = 1,p=0.7,list=FALSE)
test_set<-data_clean[test_index,]
train_set<-data_clean[-test_index,]

#knn
set.seed(1, sample.kind = "Rounding")
train_knn<-train(gender~., method="knn", preProcess=c("center","scale"),tuneLength=10,
                 trControl=trainControl(method="cv",number=10),data = train_set)
knn_predict<-predict(train_knn,test_set)
knn<-mean(knn_predict==test_set$gender) 
train_knn$results%>%ggplot(aes(k,Accuracy))+geom_line()
knn

#rpart
set.seed(1, sample.kind = "Rounding")
train_rpart<-train(gender~., method="rpart", tuneGrid=data.frame(cp=seq(0,0.05,0.002)),data = train_set)
rpart_predict<-predict(train_rpart,test_set)
rpart<-mean(rpart_predict==test_set$gender) 

plot(train_rpart$finalModel, margin = 0.1)
text(train_rpart$finalModel, cex = 0.75)
rpart

#rf
set.seed(1, sample.kind = "Rounding")
train_rf<-train(gender~.,method = "rf", tuneGrid=data.frame(mtry=seq(1,7)), ntree= 150, data=train_set)
rf_predict<-predict(train_rf,test_set)
rf<-mean(rf_predict==test_set$gender) 
rf

#importance of various predictors
plot(varImp(train_rf))


#visualize results
data.frame(Method=c("KNN","RPART","RF"), Accuracy=c(knn,rpart,rf))%>%ggplot(aes(Method, Accuracy))+
  geom_bar(stat = "identity", fill="red",width = 0.3 )+coord_flip()+ggtitle("Summary of Accuracies of Each Model")

