function [X] = ANPM4GEC(X,A,B,maxitr,innItr,q)
% tildeX_t = inv(B)*A*X_t - beta_t*X_t-1 
% X_t+1 = tildeX_t * (tildeX_t'*B*tildeX_t)^-.5

    X_old = zeros(size(X));    
    X = GSgen(X,B,0); 
    p = size(X,2);
    for t=1:maxitr
        Sigma = (X'*(B*X))\(X'*(A*X));
        diagSigma = sort(diag(Sigma),'descend');
        beta = diagSigma(q)^2/4; 
        Y = X*Sigma;  
        for j=1:p
            [Y(:,j),~] = pcg(B,A*X(:,j),[],innItr,[],[],Y(:,j));
        end               
        tmp = Y - beta*X_old;
        X_old = X;
        X = GSgen(tmp,B,0); 
    end
end

function [ U ] = GSgen( V,B,r )
    n = size(V,1);
    k = size(V,2);
    U = zeros(n,k);
    U(:,1) = V(:,1)/sqrt((V(:,1)'*B)*V(:,1)+r*(V(:,1)'*V(:,1)));
    for i = 2:k
        U(:,i) = V(:,i);
        for j = 1:i-1
           U(:,i) = U(:,i) - ( (U(:,i)'*B)*U(:,j) + r*U(:,i)'*U(:,j))*U(:,j);
        end
        U(:,i) = U(:,i)/sqrt((U(:,i)'*B)*U(:,i) + r*U(:,i)'*U(:,i));
    end
end