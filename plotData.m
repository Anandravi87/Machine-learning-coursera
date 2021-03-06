function plotData(X, y)
%PLOTDATA Plots the data points X and y into a new figure 
%   PLOTDATA(x,y) plots the data points with + for the positive examples
%   and o for the negative examples. X is assumed to be a Mx2 matrix.

% Create New Figure
figure; hold on;

% ====================== YOUR CODE HERE ======================
% Instructions: Plot the positive and negative examples on a
%               2D plot, using the option 'k+' for the positive
%               examples and 'ko' for the negative examples.
%

%First, find the indices of positive and negative values of y
pos = find(y==1);
neg = find(y==0);

%plot the positives 
plot(X(pos,1),X(pos,2), 'k+', 'markersize', 7,'markerfacecolor', 'b');
plot(X(neg,1),X(neg,2), 'ko', 'markersize', 7,'markerfacecolor', 'y');

xlabel('Exam 1 score');
ylabel('Exam 2 score');

legend('Admitted', 'Not admitted');

% =========================================================================

hold off;

end
