function [ f, g ] = fv_grad( w_curr, X, y )
%FV_GRAD: returns the objective function value f and the gradient g w.r.t w
%at point w_curr
%   w_curr is current weight vector (d * 1)
%   X is feature values (d * n)
%   y is label (1 * n)

% Set parameters
f = 0;
g = zeros(length(w_curr), 1);
lambda_val = 25;
lambda = ones(length(w_curr), 1) * lambda_val;
lambda(1) = 0;  % do not penalize weight associated with the intercept term
[row,col] = size(X);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%likelihood functon is modelled as sigmoid. probabilty = 1/(1+exp(-x.w))
dot_prod = w_curr'*X;
sig = sigmoid(dot_prod);

% separate the features for the two classes
index_zero = find(y == 0 );
index_one = find(y == 1 );
feature_one = sig(:,index_one);
feature_zero = sig(:,index_zero);

% calculate negative log likelihood for classes
f1 = -1*log(feature_one);
f0 = -1*log(1-feature_zero);

%sum across all instances 
f = sum(f1)+ sum(f0); 

%calculate gradient
g = X*(y - sig)';
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

end

