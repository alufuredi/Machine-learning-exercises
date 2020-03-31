function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;
c_test = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
sig_test = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
error_table = zeros(64,3);

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
n=1; %initialize error table counter
for i1 = c_test
    for i2 = sig_test % iterate over C and sigma
        model = svmTrain(X, y, i1, @(x1, x2) gaussianKernel(x1, x2, i2)); %i1==C, i2==sigma
        predictions = svmPredict(model, Xval);
        cost = mean(double(predictions ~= yval));%get performance per C,sigma values
        %update error table with C, sigma, cost
        error_table(n,1) = i1;
        error_table(n,2) = i2;
        error_table(n,3) = cost;
        n = n +1; % increment counter
    end
end

%find the lowest cost and its index for error table
[Y,I] = min(error_table(:,3));
C = error_table(I,1);
sigma = error_table(I,2);



% =========================================================================

end
