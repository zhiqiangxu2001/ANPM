function [Ux,Uy] = ANPM4CCA(X, Y, Ux, Uy, innerIter, outerIter, r, q)

    N = size(X,2);
    M = innerIter; 
    m = N;
    eta = 1;
    eta_x = eta/max(sum(abs(X*sqrt(N)).^2,1)); 
    eta_y = eta/max(sum(abs(Y*sqrt(N)).^2,1));

    XU = X'*Ux;     YU = Y'*Uy;
    for i=1:outerIter
        if rem(i,2)==1
            Ux_1 = Ux;
            Gt = ((XU'*XU)+r*(Ux'*Ux))\(XU'*YU);    
            Ux = SVRG_k(Ux*Gt,Uy,X*sqrt(N),Y*sqrt(N),r,M,m,eta_x);              
            XU = X'*Ux;                                                     

            Gt = ((YU'*YU)+r*(Uy'*Uy))\(YU'*XU);   
            Uy = SVRG_k(Uy*Gt,Ux,Y*sqrt(N),X*sqrt(N),r,M,m,eta_y);            
            YU = Y'*Uy;                                                      
        else            
            Gt = ((XU'*XU)+r*(Ux'*Ux))\(XU'*YU);                              
            diagGt = sort(diag(Gt),'descend');
            beta = diagGt(q)^2/4; 
            Ux_tilde = SVRG_k(Ux*Gt,Uy,X*sqrt(N),Y*sqrt(N),r,M,m,eta_x);    
            Ux1 = Ux_tilde - beta*Ux_1;
            Ux_1 = Ux;
            Ux = GSgen(Ux1,X,r);                                           
            XU = X'*Ux;                                                      

            Gt = ((YU'*YU)+r*(Uy'*Uy))\(YU'*XU);
            diagGt = sort(diag(Gt),'descend');
            beta = diagGt(q)^2/4; 
            Uy_tilde = SVRG_k(Uy*Gt,Ux,Y*sqrt(N),X*sqrt(N),r,M,m,eta_y);    
            Uy1 = Uy_tilde - beta*Uy; 
            Uy = GSgen(Uy1,Y,r);                                           
            YU = Y'*Uy;                                                     
        end
    end
end

function [ U ] = GSgen( V,B, r )
    n = size(V,1);
    k = size(V,2);
    U = zeros(n,k);
    U(:,1) = V(:,1)/sqrt(norm(B'*V(:,1))^2+r*norm(V(:,1))^2);
    for i = 2:k
        U(:,i) = V(:,i);
        for j = 1:i-1
           U(:,i) = U(:,i) - ( (U(:,i)'*B)*(B'*U(:,j)) + r*U(:,i)'*U(:,j))*U(:,j);
        end
        U(:,i) = U(:,i)/sqrt(norm(B'*U(:,i))^2 + r*norm(U(:,i))^2);
    end
end

function [ U_j ] = SVRG_k( U_j,V,X,Y,r_x,M,m,eta )
    [~,N] = size(X);
    for j=1:M
        W_0 = U_j;
        W_t = W_0;
        batch_grad = X*(X'*W_0-Y'*V)/N+r_x*W_0;
        for t=1:m
            i_t = randi([1,N]);
            x_i_t = X(:,i_t);
            W_t = W_t - eta*(x_i_t*(x_i_t'*(W_t-W_0)) + r_x*(W_t-W_0)+batch_grad);
        end
        U_j = W_t;
    end
end

