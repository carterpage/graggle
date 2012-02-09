function [theta] = train(X, y, lambda, num_iter)

% Some useful variables
m = size(X, 1);
n = size(X, 2);

% You need to return the following variables correctly 
theta = zeros(1, n + 1);

% Add ones to the X data matrix
X = [ones(m, 1) X];

options = optimset('GradObj', 'on', 'MaxIter', num_iter);
initial_theta = zeros(n + 1, 1);
[theta] = fmincg(@(t)(lrCostFunction(t, X, y, lambda)), initial_theta, options);
end
