function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


%Start Part 1: Implement the Feedforward algo and return Cost (J)

%Add a column of 1s to X - to account for the bias
X = [ones(m,1) X];

%assign a1 to the input X
a1 = X; %this is the input layer of the NN

z2 = a1*Theta1'; %dim of z2 = 5000*25

a2 = sigmoid (z2); %a2 is the activation of the hidden layer consisting of 25 units

a2 = [ones(m,1) a2]; %Add a column of 1s to a2 - to account for the bias
%dim of a2 = 5000*26

z3 = a2*Theta2';  %dim of z3 is 5000*10

a3 = sigmoid (z3); %a3 is the hypothesis of the output layer consisting of 10 units (k)

%recode the labels (y's) to contain only values 0's and 1's (creating a logical array)
%assign an empty matrix
y_matrix = [];

%creating a logical array
for k = 1:num_labels
	y_matrix = [y_matrix y==k];
end;

%Compute cost using a For loop to loop over all the training examples 
%non-regularised part

for i = 1:m
	J = J + ((log(a3(i,:))*-y_matrix(i,:)')-(log(1-a3(i,:))*(1-y_matrix(i,:))'))/m;	
end;

%Compute cost for the regularised part
J_reg = sum(sum(Theta1(:,2:input_layer_size+1).^2)) + sum(sum(Theta2(:,2:hidden_layer_size+1).^2));
J_reg = J_reg*lambda/(2*m);

%Add the Cost for the regularised part to J to get the final Cost
J = J + J_reg;

%Start Part 2: Implement the Backprop algo to calculate the Gradient
%calculate the small deltas
delta_small3 = a3 -  y_matrix; %dim of delta_small3 = 5000*10

%next step is to compute delta_small2. It should be the same dim as Theta2
delta_small2 = delta_small3*Theta2(:,2:hidden_layer_size+1) .* sigmoidGradient(z2);

%next step is to compute the capital deltas
cap_delta2 = a2'* delta_small3; %dim of cap_delta2 = 26*10. This step calculates the 
%product and sum of errors

cap_delta1 = a1'* delta_small2; %dim of cap_delta1 = 401*25. This step calculates the 
%product and sum of errors

Theta1_grad = (cap_delta1')/m;
Theta2_grad = (cap_delta2')/m;

%implementing the regularization part 
Theta1_grad(:,2:input_layer_size+1) = Theta1_grad(:,2:input_layer_size+1) + ...
								(lambda/m)*Theta1(:,2:input_layer_size+1);
Theta2_grad(:,2:hidden_layer_size+1) = Theta2_grad(:,2:hidden_layer_size+1) + ...
								(lambda/m)*Theta2(:,2:hidden_layer_size+1);

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
