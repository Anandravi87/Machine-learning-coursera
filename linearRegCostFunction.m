function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

%Start

%Calculate h; dim(theta) = 2*1; dim(h) = 12*1; dim(X) = 12*2
h = X*theta;

%Calculate the un-regularised and regularised J separately ...
%and add them up
J_un_reg = sum(((h-y).^2)/(2*m));
theta_len = length(theta);
J_reg = (lambda/(2*m))*sum((theta(2:theta_len).^2));
J= J_un_reg+ J_reg;

%Calculate gradients
%unregularised part of the Gradient; dim(grad_un_reg) =...
%1*1
grad(1,1) = (1/m)*X(:,1)'*(h-y);

%regularised part of the Gradient
grad(2:theta_len,1) = (1/m)*X(:,2:theta_len)'*(h-y) + (lambda/m)*theta(2:theta_len);

% =========================================================================

grad = grad(:);

end
