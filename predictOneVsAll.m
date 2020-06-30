function p = predictOneVsAll(all_theta, X)
%PREDICT Predict the label for a trained one-vs-all classifier. The labels 
%are in the range 1..K, where K = size(all_theta, 1). 
%  p = PREDICTONEVSALL(all_theta, X) will return a vector of predictions
%  for each example in the matrix X. Note that X contains the examples in
%  rows. all_theta is a matrix where the i-th row is a trained logistic
%  regression theta vector for the i-th class. You should set p to a vector
%  of values from 1..K (e.g., p = [1; 3; 1; 2] predicts classes 1, 3, 1, 2
%  for 4 examples) 

m = size(X, 1);
num_labels = size(all_theta, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% Add ones to the X data matrix
X = [ones(m, 1) X];

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned logistic regression parameters (one-vs-all).
%               You should set p to a vector of predictions (from 1 to
%               num_labels).
%
% Hint: This code can be done all vectorized using the max function.
%       In particular, the max function can also return the index of the 
%       max element, for more information see 'help max'. If your examples 
%       are in rows, then, you can use max(A, [], 2) to obtain the max 
%       for each row.
%       

%start
%useful to note the dimensions of the matrices to begin with
%all_theta: num_labels * number of features + 1
%X: number of examples (ie 5000) * number of features + 1
%p: number of examples (ie 5000) * 1

%step 1: compute h
%h should be a dimension of number of examples (ie 5000) * num_labels+1; I need to later figure out which one out of the 
%num_labels outputs is the right one (basically by taking max)

h = sigmoid(X*all_theta');

%h will have 5000 rows and 10 cols

%calculate the max value row-wise; capture the index of the max value in the var idx
%idx should be a vector with size 5000 
[a,idx] = max(h, [], 2);

%assign the output vector to idx
p = idx;

% =========================================================================


end
