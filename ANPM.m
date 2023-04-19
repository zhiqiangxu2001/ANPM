function [X] = ANPM(X,A,nVar,maxItr,q)
% X - n x p matrix iterate
% A - n x n input matrix
% nVar - noise variance at each step
% maxItr - maximum number of iterations
% q - intermediate rank
   
    X = orth(X);
    X_old = zeros(size(X));       
    for t=1:maxItr
        Noise = random('normal',0,nVar(t),size(X));
        Y = A*X + Noise;
        Sigma = (X'*X)\(X'*Y);
        diagSigma = sort(diag(Sigma),'descend');
        beta = diagSigma(q)^2/4; 
        tmp = Y - beta*X_old;
        X_old = X;
        X = tmp/((tmp'*tmp)^.5);
    end
end