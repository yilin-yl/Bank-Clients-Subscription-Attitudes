##Data preprocessing
library(readr)
d <- read_csv("bank3.csv")
d = subset(d,job!=12)
d = subset(d,education!=4)
d = subset(d,contact!=0)
d$contact=ifelse(d$contact == 2,0,1)
d = data.frame(scale(d[,-17]),d[,17])   

library(caret)
set.seed(6)
index1 = createDataPartition(d$y,p=0.75,list=FALSE)
table(d$y)
prop.table(table(d$y))
train = d[index1,]
train_2 = d[index1,]
test = d[-index1,]
index2 = createDataPartition(train$y,p=0.6,list=FALSE)
valid = train[-index2,]
train = train[index2,]

#BEFORE SMOTE
lbls=c(0,1)
barchart(prop.table(table(as.numeric(train$y))),labels=lbls,main='y condition',col=c("lightsteelblue3",'white'),horizontal=FALSE)
prop.table(table(as.numeric(train$y)))
table(train$y)

#SMOTE
library(smotefamily)
train$y = as.factor(train$y)
train = BLSMOTE(train[,-17],train$y,K=10,C=5,method='type1')
train = train$data
table(train$class)
names(train)[17] = 'y'
train = na.omit(train)

#AFTER SMOTE
barchart(prop.table(table(as.numeric(train$y))),labels=lbls,main='y condition',col=c("lightsteelblue3",'white'),horizontal=FALSE)
prop.table(table(as.numeric(train$y)))
table(train$y)


library("corrplot")
cor_data=cor(d)
corrplot(corr=cor_data)

library(GGally)
ggpairs(d)




###Logistic Regression
train$y = as.numeric(train$y)
glm.fits=glm(y~.,data=train,family=binomial)  
summary(glm.fits)
glm.probs=predict(glm.fits, valid, type="response")
glm.pred=rep(0, nrow(valid))     
glm.pred[glm.probs>.5]=1       
table(glm.pred, valid$y)
mean(glm.pred==valid$y)   
library(gmodels)
CrossTable(glm.pred, valid$y)

###feature selection
#####backward
library(MASS)
stepAIC(glm.fits, direction = 'backward')
glm.fits2=glm(y ~ age + education + default + balance + housing + loan + month + duration + campaign + pdays + previous + poutcome, family = binomial, data = train)  
summary(glm.fits2)
glm.probs2=predict(glm.fits2, valid, type="response")
glm.pred2=rep(0, nrow(valid))     
glm.pred2[glm.probs2>.5]=1   
table(glm.pred2, valid$y)
mean(glm.pred2==valid$y)  

#####R^2 & Cp
library(leaps)
exps<-regsubsets(y~age+job+marital+education+default+balance+housing+loan+contact+day+month+duration+campaign+pdays+previous+poutcome, data=d, nbest=3, really.big=T)
expres<-summary(exps)
res<-data.frame(expres$outmat, adjustedR^2=expres$adjr2)
res2<-data.frame(expres$outmat, Cp<-expres$cp)

glm.fits3=glm(y ~ age + education + housing + loan + duration + campaign + previous + poutcome,data=train,family=binomial)  
summary(glm.fits3)
glm.probs3=predict(glm.fits3, valid, type="response")
glm.pred3=rep(0, nrow(valid))     
glm.pred3[glm.probs3>.5]=1   
table(glm.pred3, valid$y)
mean(glm.pred3==valid$y)   

####LASSO
library(Matrix)
library(foreach)
library(glmnet)
x<-as.matrix(train[,-17])
y<-train[,17]
cvfit<-cv.glmnet(x, y, family='binomial', alpha=1, type.measure='mse')
plot(cvfit)
coef(cvfit, s='lambda.1se')
b<-glmnet(x, y, alpha=1, family='binomial', lambda=cvfit$lambda)
plot(b, xvar='lambda', label=TRUE)
out<-glmnet(x, y, alpha=1, family='binomial')
testx<-as.matrix(valid[,-17])
lasso.pred0<-predict(cvfit, newx=x, type='class', s='lambda.min')
lasso.pred1<-predict(cvfit, newx=testx, type='class',s='lambda.1se')
ST0<-y
ST1<-valid[,17]
table(ST0, lasso.pred0)
table(ST1, lasso.pred1)
mean(ST1==lasso.pred1)        


###LDA       
library(MASS)
lda.fit=lda(y~.,data=train)   
lda.fit
plot(lda.fit)
lda.pred=predict(lda.fit, valid)
lda.class=lda.pred$class   
table(lda.class,valid$y)     
mean(lda.class==valid$y)    


###QDA    
qda.fit<-qda(y~.,data=train)
qda.fit
qda.class<-predict(qda.fit, valid)$class
table(qda.class,valid$y)
mean(qda.class==valid$y)     


###QDA with feature selection
qda.fit3<-qda(y~age + education + housing + loan + duration + campaign + previous + poutcome, data=train)
qda.class3<-predict(qda.fit3, valid)$class
table(qda.class3,valid$y)
mean(qda.class3==valid$y)  

##KNN
c <- dim(valid)[1]
c
train$y = as.numeric(train$y)
library('kknn')
for (n in 4:15){
  dkknn=kknn(y~.,train,valid,distance=2,kernel="triangular",k=n)
  fit=fitted(dkknn)
  table(fit,test$y)
  
  t=0.6
  for (j in 1:919){
    if(fit[j]>=t){
      fit[j]=1
    }
    else{
      fit[j]=0
    }
  }
  num=0
  for (m in 1:919){
    if(fit[m]==test$y[m]){
      num = num + 1
    }
  }
  accuracy=num/919
  print(n)
  print(t)
  print(accuracy)
}
table(fit, valid$y)

#Support Vector Machine
#kernel linear
library(e1071)
set.seed(3)
tune.out = tune(svm,y~.,data=train,kernel='linear',
                ranges=list(cost=c(0.001,0.01,0.1,1,5,10,100)))
summary(tune.out)
summary_cvsvm = summary(tune.out)
summary_cvsvm$best.parameters
svmfit = svm(y~.,data=train,kernel='linear',cost=5)
summary(svmfit)
svm.pred = predict(svmfit,valid)
table(svm.pred,valid$y)
mean(svm.pred==valid$y) #0.7834603

#kernel radial
set.seed(3)
tune.out2 = tune(svm,y~.,data=train,kernel='radial',
                 ranges=list(cost=c(0.1,1,10,100,1000),gamma=c(0.5,1,2,3,4)))
summary_cvsvm2 = summary(tune.out2)
summary_cvsvm2$best.parameters
summary_cvsvm2
svmfit2 = svm(y~.,data=train,kernel='radial',cost=1,gamma=0.5)
summary(svmfit2)
svm.pred2 = predict(svmfit2,valid)
table(svm.pred2,valid$y)
mean(svm.pred2==valid$y) #0.8422198

#kernel poly
set.seed(3)
tune.out3 = tune(svm,y~.,data=train,kernel='polynomial',
                 ranges=list(cost=c(0.1,1,10,100,1000),degree=c(1,2,3,4)))
summary_cvsvm3 = summary(tune.out3)
summary_cvsvm3$best.parameters
svmfit3 = svm(y~.,data=train,kernel='polynomial',cost=10,degree=3)
summary(svmfit3)
svm.pred3 = predict(svmfit3,valid)
table(svm.pred3,valid$y)
mean(svm.pred3==valid$y) #0.7986942


#decision tree
library(tree)
train$y = as.factor(train$y)
valid$y = as.factor(valid$y)
tree.fit = tree(y~., train)
summary(tree.fit)
set.seed(100)
cv.y = cv.tree(tree.fit,FUN=prune.misclass)
cv.y
plot(cv.y$size,cv.y$dev,type='b')
tree.pred = predict(tree.fit,valid,type="class")
plot(tree.fit)
text(tree.fit,pretty = 0)
table(tree.pred,valid$y)
mean(tree.pred == valid$y) #0.7595212

#Random Forest
set.seed(66)
library(randomForest)
vscore = rep(0,6)
for (i in 1:6){
  vrf = randomForest(y~.,mtry=i,data=train)
  vpred = predict(vrf,valid)
  vscore[i] = mean(vpred == valid$y)
}
vscore
which.max(vscore)
set.seed(66)
rf.fit = randomForest(y~.,mtry=2,data = train,importance=TRUE)
rf.pred = predict(rf.fit,valid)
importance(rf.fit,type=1)
par(mfrow=c(1,2))
varImpPlot(rf.fit)
table(rf.pred,valid$y)
mean(rf.pred==valid$y) #0.8563656


library(gbm)
library(ISLR)
library(ipred)
library(caretEnsemble)
library(xgboost)
library(mlr)
library(sets)
library(XML)
library(pROC)
library(DiagrammeR)
library(emoa)


##SMOTE for train_2
table(train_2$y)
train_2$y = as.factor(train_2$y)
train_2 = BLSMOTE(train_2[,-17],train_2$y,K=10,C=5,method='type1')
train_2 = train_2$data
table(train_2$class)
names(train_2)[17] = 'y'



##Tree Bagging
####Model Training
set.seed(6)
train$y = factor(train$y)
index = seq(10,300,1)
accu = c()
for (i in 10:300){
  bank_bag_tree = bagging(y~.,data=train,coob=T,nbagg=i)
  bagging_pred<-predict(bank_bag_tree,valid)
  y = mean(bagging_pred==valid$y)
  accu <- c(accu,y)
}
plot(index,accu,type='l')
max(accu)
points(25,max(accu),pch=20)
summary(bank_bag_tree$mtrees)
bagging_pred<-predict(bank_bag_tree,valid)
mean(bagging_pred==valid$y)
CrossTable(bagging_pred,valid$y)

####Model Evalution
bagging_roc <- roc(valid$y,as.numeric(bagging_pred))
plot(bagging_roc, print.auc=TRUE, auc.polygon=TRUE, 
     grid=c(0.1, 0.2),grid.col=c("green", "red"), max.auc.polygon=TRUE,
     auc.polygon.col="skyblue", print.thres=TRUE,main='bagging ROC')



##XGBoost
####Tuning Parameters
fact_col <- as.character(colnames(train))
for(i in fact_col) set(train,j=i,value = factor(train[[i]]))
for (i in fact_col) set(valid,j=i,value = factor(valid[[i]]))
for (i in fact_col) set(test,j=i,value = factor(test[[i]]))

traintask <- makeClassifTask (data = train,target = "y")
valid$y = factor(valid$y)
validtask <- makeClassifTask (data = valid,target = "y")
traintask <- createDummyFeatures (obj = traintask) 
validtask <- createDummyFeatures (obj = validtask)
test$y = factor(test$y)
testtask <- makeClassifTask (data = test,target = "y")
testtask <- createDummyFeatures (obj = testtask)
lrn <- makeLearner("classif.xgboost",predict.type = "response")
lrn$par.vals <- list( objective="binary:logistic", eval_metric="error", 
                      nrounds=200L)
params <- makeParamSet( makeDiscreteParam("booster",values = "gbtree"), 
                        makeNumericParam("eta",lower = 0L,upper = 1L),
                        makeNumericParam("subsample",lower = 0L,upper = 1L),
                        makeIntegerParam('max_depth',lower=1L,upper=10L),
                        makeNumericParam("colsample_bytree",lower = 0L,upper = 1L),
                        makeIntegerParam('min_child_weight',lower=0L,upper=10L))
rdesc <- makeResampleDesc("CV",stratify = T,iters=3L)
ctrl <- makeTuneControlGrid(resolution=10L)
set.seed(6)
mytune <- tuneParams(learner = lrn, task = traintask, 
                     resampling = rdesc, measures = acc, par.set = params, 
                     control = ctrl)
mytune

####Model Training
set.seed(6)
train_dat = data.matrix(train[,c(-17)])
valid_dat = data.matrix(valid[,-17])
test_dat = data.matrix(test[,-17])
train_label = as.numeric(train$y)
valid_label = as.numeric(valid$y)
test_label= as.numeric(test$y)
xgb_train = xgb.DMatrix(data=train_dat,label=train_label)
xgb_valid = xgb.DMatrix(data=valid_dat,label=valid_label)
xgb_test = xgb.DMatrix(data=test_dat,label=test_label)
xgb_model = xgboost(data = xgb_train, nrounds=200,eta=0.331,max.depth=2,
                    objective = "binary:logistic")
xgb_pred1 = predict(xgb_model,xgb_valid)
xgb_pred1[xgb_pred1>0.5] = 1
xgb_pred1[xgb_pred1<=0.5] = 0
mean(xgb_pred1==valid$y)

####Final Model 
train_best = data.matrix(train_2[,c(-17)])
best_label = as.numeric(train_2$y)
xgb_whole = xgb.DMatrix(data=train_best,label=best_label)
xgb_best = xgboost(data = xgb_whole, nrounds=200,eta=0.331,max.depth=2,
                   objective = "binary:logistic")
xgb_pred2 = predict(xgb_best,xgb_test)
xgb_pred2[xgb_pred2>0.5] = 1
xgb_pred2[xgb_pred2<=0.5] = 0
mean(xgb_pred2==test$y)

####Final Model Evaluation

######Importance Matrix
names = as.character(dimnames(train)[[2]])
importance_matrix = xgb.importance(names,model=xgb_model)
xgb.plot.importance(importance_matrix,cex=0.8,xlim=c(0,0.5))

######Tree0
xgb.plot.tree(model = xgb_model,trees=0)
confusionMatrix(factor(xgb_pred2),factor(test$y))

######ROC Curve
xgboost_roc <- roc(test$y,as.numeric(xgb_pred2))
plot(xgboost_roc, print.auc=TRUE, auc.polygon=TRUE, 
     grid=c(0.1, 0.2), max.auc.polygon=TRUE,
     print.thres=TRUE,main='xgboost ROC')
