function X_norm = featureNormalize(X,mu,sigma)
%FEATURENORMALIZE Normalizes the features in X 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.

% You need to set these values correctly
X_norm = X;

% ====================== YOUR CODE HERE ======================
% Instructions:  
%               Note that X is a matrix where each column is a 
%               feature and each row is an example. You need 
%               to perform the normalization separately for 
%               each feature. 
%
% 
%       
for i = 1: size(X_norm,2),
    for j = 1:size(X_norm,1),
        X_norm(j,i) = (X_norm(j,i)- mu(i))/sigma(i);
    end;
end;

% ============================================================

end
