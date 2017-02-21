function [mu,sigma] = getNormalizeParameter(X)

% ====================== YOUR CODE HERE ======================
% Instructions: First, for each feature dimension, compute the mean
%               of the feature and subtract it from the dataset,
%               storing the mean value in mu. Next, compute the 
%               standard deviation of each feature and divide
%               each feature by it's standard deviation, storing
%               the standard deviation in sigma. 
% Hint: You might find the 'mean' and 'std' functions useful.

mu = mean(X);
sigma = std(X);

end
