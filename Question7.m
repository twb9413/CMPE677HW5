b = load('bupa.data');
X = b(:,1:end-1);
y = b(:,end); 
 
options.method = 'Adaboost';
options.numberOfFolds = 5;
options.adaboost_numFeatures=500;
[confusionMatrix,accuracy] =  classify677_hwk5(X,y,options);
