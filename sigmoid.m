function g = sigmoid(z)
%SIGMOID Compute sigmoid function
%   g = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 
g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).

row_size = size(z,1);
col_size = size(z,2);

for i = 1:row_size
	for j = 1:col_size
		g(i,j) = 1/(1+exp(-z(i,j)));
	end
end

% =============================================================

end
