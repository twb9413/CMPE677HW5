% load the bupa data
% cd to location where hwk5 files are
cd C:\Users\tbrad\Documents\MATLAB\CMPE677HW5\
% add path to where hwk4 files are
addpath C:\Users\tbrad\Documents\MATLAB\CMPE677HW4\
close all ; clear all;
b = load('bupa.data');
X = b(:,1:end-1);
y = b(:,end); 
 
options.method = 'LogisticRegression';
options.numberOfFolds = 5;
options.lambda = 0.1;
[confusionMatrix,accuracy] =  classify677_hwk5(X,y,options);

%-------------------------------------------------------------------------
% test 6b
% develop adaboost
close all ; clear all;
b = load('bupa.data');
Xdata = b(:,1:end-1);
y = b(:,end); 
 
%turn ground truth labels into {-1,+1}
yList = unique(y);
if yList(1) ~= -1
    y(y==yList(1))=-1;
    y(y==yList(2))= 1;
end
         
%form train and test sets
TrainXdata = Xdata(1:200,:);
TrainGT = y(1:200);
TestXdata = Xdata(201:end,:);
TestGT = y(201:end);
 
%number of features
adaboost_numFeatures=500;
 
%for every feature, find the best threshold
D = size(TrainXdata,2);  %number of features , fn
n = size(TrainXdata,1);  %number of samples, fs
nlist = [1:n]'; 
 
%Each sample gets a weight r
weights = TrainGT.*0+1./n;   %equal weights to start
 
 %each iteration creates a classifier. 
h1=decisionStumpW(weights,TrainXdata,TrainGT)    ;                %train base learner
[errorAmt, alpha] = AdaBoostError(weights, h1,TrainXdata, TrainGT); %determine alpha
disp(errorAmt)
disp(alpha)
%errorAmt = 0.3750, alpha = 0.2554    

%--------------------------------------------------------------------------
% test 6c
h1.alpha = alpha;
 
[newWeights,zz] = AdaBoostUpdateWeights(weights,h1,TrainXdata,TrainGT);    %update the weights
% newWeights(1:5)
%     0.0067
%     0.0040
%     0.0040
%     0.0040
%     0.0067
%zz = 0.9682   

%---------------------------------------------------------------------------
% test 6d

close all ; clear all;
b = load('bupa.data');
Xdata = b(:,1:end-1);
y = b(:,end); 
 
%turn ground truth labels into {-1,+1}
yList = unique(y);
if yList(1) ~= -1
    y(y==yList(1))=-1;
    y(y==yList(2))= 1;
end
         
%form train and test sets
TrainXdata = Xdata(1:200,:);
TrainGT = y(1:200);
TestXdata = Xdata(201:end,:);
TestGT = y(201:end);
 
%number of features
adaboost_numFeatures=500;
[classifiers, errors, pred] = myAdaBoost(TrainXdata,TrainGT,adaboost_numFeatures,TestXdata,TestGT);
            
meanTrain = errors.train;
meanTest = errors.test;
meanEB = errors.eb;
 
 
figure 
hold on
x = 1:1:adaboost_numFeatures; 
plot(x, meanEB,'k:',x,meanTest,'r-',x,meanTrain,'b--','LineWidth',2); 
legend('ErrorBound','TestErr','TrainErr','Location','Best');
xlabel 'iteration (number of Classifiers)'
ylabel 'error rates (50 trials)'
title 'AdaBoost Performance on Bupa'
print -dpng hwk5_10e.png


