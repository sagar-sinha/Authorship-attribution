function K = kernel_intersection(X, X2)

n = size(X,1);
m = size(X2,1);
l = size(X,2);
K = zeros(m, n);
X1 = transpose(X);
X2 = transpose(X2);
for i = 1:m
    for j = 1:n
        sum1 = sum(min(X2(:,i),X1(:,j)));
        K(i,j) = sum1;
    end
end
