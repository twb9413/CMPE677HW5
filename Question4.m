% HW Question:  Complete the two new methods to the sample classification 
% code below for the cases ‘ClassificationTree’, and ‘BaggedTree’ using 
% the parameters as above (and make sure you use rng(2000) for BaggedTree).
% Report the accuracy and confusion matrix for each.  

clear ; close all; clc
 
% Load Training Data- Andrew Ng Machine Learning MOOC
load('ex3data1.mat'); % training data stored in arrays X, y
n = size(X, 1);
num_labels =  length(unique(y));          % 10 labels, from 1 to 10   (note  "0" is mapped to label 10)
 
% Randomly select 100 data points to display
rng(2000);  %random number generator seed
rand_indices = randperm(n);
sel = X(rand_indices(1:100), :);
 
%displayData(sel);
%print -djpeg95 hwk4_4.jpg
 
Xdata = [ones(n, 1) X];
% the matlab functions you want to use are crossvalind.m and confusionmat.m_
% Xdata- A vector of feature, nxD, one set of attributes for each dataset sample
% y- A vector of ground truth labels, nx1 (each class has a unique integer value), one label for each dataset sample
% numberOfFolds- the number of folds for k-fold cross validation
numberOfFolds=5;
rng(2000);  %random number generator seed
CVindex = crossvalind('Kfold',y, numberOfFolds);
 
method='BaggedTree'
 
lambda = 0.1;
for i = 1:numberOfFolds
    TestIndex = find(CVindex == i);
    TrainIndex = find(CVindex ~= i);
    
    TrainDataCV = Xdata(TrainIndex,:);
    TrainDataGT =y(TrainIndex);
    
    TestDataCV = Xdata(TestIndex,:);
    TestDataGT = y(TestIndex);
    
    %
    %build the model using TrainDataCV and TrainDataGT
    %test the built model using TestDataCV
    %
    switch method
        case 'LogisticRegression'
            % for Logistic Regression, we need to solve for theta
            % Initialize fitting parameters
            all_theta = zeros(num_labels, size(Xdata, 2));
            
            for c=1:num_labels
                % Set Initial theta
                initial_theta = zeros(size(Xdata, 2), 1);
                % Set options for fminunc
                options = optimset('GradObj', 'on', 'MaxIter', 50);
                
                % Run fmincg to obtain the optimal theta
                % This function will return theta and the cost
                [theta] = ...
                    fmincg (@(t)(costFunctionLogisticRegression(t, TrainDataCV, (TrainDataGT == c), lambda)), ...
                    initial_theta, options);
                
                all_theta(c,:) = theta;
            end

            % Using TestDataCV, compute testing set prediction using
            % the model created
            % for Logistic Regression, the model is theta
            % Insert code here to see how well theta works...
            all_pred = sigmoid(TestDataCV*all_theta');
            [maxVal,maxIndex] = max(all_pred,[],2);
            TestDataPred=maxIndex;
            
        case 'KNN'
            [idx, dist] = knnsearch(TrainDataCV,TestDataCV,'k',3);
            TestDataPred=mode([TrainDataGT(idx(:,1)) TrainDataGT(idx(:,2))  TrainDataGT(idx(:,3)) ]')';
            
         case 'ClassificationTree'
            tree = ClassificationTree.fit(TrainDataCV, TrainDataGT);
            maxprune = max(tree.PruneList);
            treePrune = prune(tree,'level',maxprune-3);
            TestDataPred = predict(tree,TestDataCV);
            
         case 'BaggedTree'
            rng(2000);  %random number generator seed
            t = ClassificationTree.template('MinLeaf',1);
            bagtree = fitensemble(TrainDataCV,TrainDataGT,'Bag',10,t,'type','classification');
            TestDataPred = predict(bagtree,TestDataCV);  %really should test with a test set here   
             
        otherwise
            error('Unknown classification method')
    end
    
    predictionLabels(TestIndex,:) =double(TestDataPred);
end
 
confusionMatrix = confusionmat(y,predictionLabels);
accuracy = sum(diag(confusionMatrix))/sum(sum(confusionMatrix));
 
fprintf(sprintf('%s: Lambda = %d, Accuracy = %6.2f%%%% \n',method, lambda,accuracy*100));
fprintf('Confusion Matrix:\n');
[r c] = size(confusionMatrix);
for i=1:r
    for j=1:r
        fprintf('%6d ',confusionMatrix(i,j));
    end
    fprintf('\n');
end

