function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

%compute the value of the Cost function J
%formula of cost function J = (1/m)*(-y'*log(h)-(1-y)'*log(1-h)) + (lambda/2*m)*theta^2
%step 1: find h; output should be m*1
h = sigmoid(X*theta);

%step 2: plug h into the partial formula of J to get J_interim_1; output should be a scalar
J_interim_1 = (1/m)*(-y'*log(h)-(1-y)'*log(1-h));

%step 3: calculate the regularization param to get J_interim_2; output is scalar
theta_len = length(theta);
J_interim_2 = (lambda/(2*m))*sum(theta(2:theta_len).^2);

%step 4: Add both the J interims to get the final J
J = J_interim_1 + J_interim_2;

%find the gradient using the below formula; output should have the dim (n+1)*1
%gradient = (1/m)*X'*(h-y)
%calculating the gradient of the first element in the gradient vector separately
grad (1,1) = (1/m)*X(:,1)'*(h-y);

%save the length of the gradient vector
grad_len = length(grad);

%calculating the subsequent	elements in the gradient vector with the regularization term 
grad (2:grad_len,1) = (1/m)*X(:,2:grad_len)'*(h-y) + (lambda/m)*theta(2:grad_len);

% =============================================================

end
