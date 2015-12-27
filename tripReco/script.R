require(xgboost)
require(methods)
require(data.table)
require(magrittr)
require(bit64)

loss_function <- function(y, pred){
  total = 0.
  for (i in 0:length(y)){
    p = max(min(pred[i][y[i]], (1 - eps)), eps)
    total = total + log(p)
  }
  return -(total/length(y))
}

train <- fread('train.csv', header = T, colClasses = "character")

train$WeekdayN <- as.numeric(as.factor(train$Weekday))
train$DepartmentDescriptionN <- as.numeric(as.factor(train$DepartmentDescription))
train$UpcN <- as.numeric(as.factor(train$Upc))
train$FinelineNumberN <- as.numeric(as.factor(train$FinelineNumber))
train$VisitNumberN <- as.numeric(as.factor(train$VisitNumber))

classNames <- paste("TripType_", unique(train$TripType), sep="");
train$TripTypeN <- as.numeric(as.factor(train$TripType)) 
train$TripTypeN <- train$TripTypeN -1

train$ScanCountN <- as.numeric(train$ScanCount)
#TripType VisitNumber Weekday         Upc ScanCount DepartmentDescription FinelineNumber 
#WeekdayN DepartmentDescriptionN  UpcN FinelineNumberN VisitNumberN TripTypeN ScanCountN
train[, TripType := NULL]
train[, VisitNumber := NULL]
train[, Weekday := NULL]
train[, Upc := NULL]
train[, ScanCount := NULL]
train[, DepartmentDescription := NULL]
train[, FinelineNumber := NULL]

trainMatrix <- train[,lapply(.SD,as.numeric)] %>% as.matrix

numberOfClasses <- max(train$TripTypeN) + 1

param <- list("objective" = "multi:softprob",
              "eval_metric" = "mlogloss",
              "num_class" = numberOfClasses)



#=================test

test <- fread('test.csv', header=T, colClasses = "character")

test$WeekdayN <- as.numeric(as.factor(test$Weekday))
test$DepartmentDescriptionN <- as.numeric(as.factor(test$DepartmentDescription))
test$UpcN <- as.numeric(as.factor(test$Upc))
test$FinelineNumberN <- as.numeric(as.factor(test$FinelineNumber))
test$VisitNumberN <- as.numeric(as.factor(test$VisitNumber))

test$ScanCountN <- as.numeric(test$ScanCount)
#TripType VisitNumber Weekday         Upc ScanCount DepartmentDescription FinelineNumber 
#WeekdayN DepartmentDescriptionN  UpcN FinelineNumberN VisitNumberN TripTypeN ScanCountN

VisitNumber <- test$VisitNumber

test[, VisitNumber := NULL]
test[, Weekday := NULL]
test[, Upc := NULL]
test[, ScanCount := NULL]
test[, DepartmentDescription := NULL]
test[, FinelineNumber := NULL]

  
testMatrix <- test[,lapply(.SD,as.numeric)] %>% as.matrix

#==== run

nround <- 512
 

bst = xgboost(param=param, data = trainMatrix, label = train$TripTypeN, nrounds=nround, max_depth=7, max_features=20,
              learning_rate=0.03, n_estimators=700 )

nfold <- 5
#bst = xgb.cv(param=param, data = trainMatrix, label = train$TripTypeN, nfold = nfold, nrounds=nround)


pred <- predict(bst, testMatrix)

predData <- data.frame(matrix(pred, ncol=numberOfClasses, byrow=T));
colnames(predData) <- classNames
res <- data.frame(VisitNumber, predData)

res_avg <- aggregate(. ~ VisitNumber, res, mean)

write.csv(res_avg, 'submission_avg3_r.csv', quote=F, row.names=F)

# Get the feature real names
names <- dimnames(trainMatrix)[[2]]


# Compute feature importance matrix
importance_matrix <- xgb.importance(names[!names %in% c('TripTypeN',  'ScanCountN')], model = bst)

# Nice graph
xgb.plot.importance(importance_matrix)

xgb.plot.tree(feature_names = names[names != 'TripTypeN'], model = bst, n_first_tree = 2)