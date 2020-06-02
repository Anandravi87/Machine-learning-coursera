function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %

	h = X*theta; %this is the hypothesis function
	errors_vec = h - y; % this is the hypothesis function minus the labels
	gradient_inter = X' * errors_vec; %need to multiply with the transpose of X to facilitate 								%matrix multiplication 
	gradient_final = gradient_inter*alpha/m; %multiply with learning rate; divide with m
	
	theta = theta - gradient_final; %calculate new theta


    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter,1) = computeCostMulti(X, y, theta);


end
