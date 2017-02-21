function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 

grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

for j = 1:length(theta),
    if j==1
        grad(j) = sum((sigmoid(X * theta) - y).* X(:,j))/m;
    else
        a = sum((sigmoid(X * theta) - y).* X(:,j))/m;
        grad(j) = a + lambda * theta(j)/m;
    end
        
end

%compute J 
a = -y .* log(sigmoid(X * theta));
b = (1 - y) .* log(1 - sigmoid(X * theta));
J = sum((a - b))/ m + lambda/(2*m)*((theta(2:end,1))'*theta(2:end,1));

% =============================================================

end
