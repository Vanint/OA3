function [classifier, summ, sen, spe, acc, que, run_time, TMs, P,idx] = OA3_sum(Y, X, options, id_list)
% OA3_sum: Online Adaptive Asymmetric Active Learning 
%---------------------------------------------------------
% Input:
%    Y:  the vector of lables
%    X:  the instance matrix
%    id_list:  a randomized ID list
%    options:  a struct containing B, rho, alpha_p, alpha_n, delta_p, delta_n, tick, eta, gamma, etc;
% Output:
%    summ:  weighted sum of sensitivity and specificity
%    sen:  sensitivity
%    spe:  specificity
%    acc:  accuracy
%    que:  the ratio of queries of labels 
%    run_time:  time consumed by this algorithm once
%    TMs:  a vector of time corresponding to mod(t,t_tick)==0
%    P:  the development of sum corresponding to mod(t,t_tick)==0
%    idx: the development of sample index corresponding to mod(t,t_tick)==0
%    classifier:  a struct containing vector w
%------------------------------------------------------------
%% initialize parameters
B = options.B;        % budgets
alpha_p = options.alpha_p;       % positive tradeoff for weighted sum
alpha_n = options.alpha_n;       % negative parameter for weighted sum 
rho = options.rho;    % \rho
rho_m = max(1,rho);   % \rho_max
delta_p = options.delta_p;    % positive query parameters
delta_n = options.delta_n;    % negative query parameters
tick     = options.tick;       % batch_size
eta = options.eta;    % learning rate
gamma = options.gamma; % regularized parameters

ID = id_list; 
t_p = 0;      % number of positive samples 
t_n = 0;      % number of negative samples 
err_p = 0;    % err number of positive samples 
err_n = 0;    % err number of negative samples 
err   = 0;    % err number
P   = [];     % the development of sum  
idx   = [];   
TMs = [];    

w=zeros(1,size(X,2));    % initialization of predictive weight
d = options.d;  % the dimensions of data
covariance = eye(d);     % initialization of covariance matrix
b = 0;   % the number of queried samples, used to calculate query ratio          

%% loop
tic   
for t = 1:length(ID),
    id = ID(t);       
    x_t = X(id,:); 

    %% prediction
    p_t=w*x_t';      
    hat_y_t = sign(p_t);   
    if (hat_y_t==0)        
        hat_y_t=1;
    end
    
    %% update t_p, t_n, err_p, err_n and err
    if (hat_y_t~=Y(id)),    
        err=err+1;         
        if Y(id)==+1;      
            err_p = err_p + 1;
        else
            err_n =err_n+1;
        end
    end
    
    if Y(id)==+1
        t_p = t_p + 1; 
    else
        t_n =t_n+1;  
    end
    
    %% query decision
    Z_t=0; % Z_t should be initialzed every rounds
    if b < B
        v_t = x_t * covariance * x_t';         % x\Sigma x
        c_t = -(eta*v_t*gamma*rho_m)/(2*v_t + 2*gamma);      
        q_t = abs(p_t)+c_t;                  % q_t = |p_t| + c_t
        if q_t <=0
            Z_t = 1;
        else
           if p_t>=0
                pr = delta_p/(delta_p+q_t);                 % query probability
                if rand <= pr
                    Z_t = 1;
                end
           else
                pr = delta_n/(delta_n+q_t);                 % query probability
                if rand <= pr
                    Z_t = 1;
                end
            end  
        end 
    end
    
    %% if query and update
    if Z_t ==1        
        b = b+1;       % queried number
        if Y(id)==+1;    
            if( p_t*Y(id) < 1)       
                covariance = covariance - [(covariance * x_t' * x_t * covariance') / (gamma+ x_t * covariance * x_t')]; 
                w = w + rho*eta* Y(id) * x_t * covariance; 
                % diagonal version
                % covariance = covariance - [(covariance .* x_t' .* x_t .* covariance') ./ (gamma+ x_t .* covariance .* x_t')];                 
                % w = w + rho*eta* Y(id) * x_t .* diag(covariance)';   
            end 
        else
            if( p_t*Y(id) < 1)  
                covariance = covariance - [(covariance * x_t' * x_t * covariance') / (gamma+ x_t * covariance * x_t')]; 
                w = w + eta* Y(id) * x_t * covariance;  
                % diagonal version
                % covariance = covariance - [(covariance .* x_t' .* x_t .* covariance') ./ (gamma+ x_t .* covariance .* x_t')];                  
                % w = w + eta* Y(id) * x_t .* diag(covariance)';  
            end
        end
    end
    
    run_time=toc;   
    % record time and sum development
    if (mod(t,tick)==0)       
        if t_p>0              
            term_p=alpha_p*(1-err_p/t_p);   
        else
            term_p=0;        
        end

        if t_n>0
            term_n=alpha_n*(1-err_n/t_n);    
        else
            term_n=0;      
        end

        P = [P term_p+term_n];  
        idx = [idx t];
        TMs=[TMs run_time];    
    end
end
acc=1-err/(t_p+t_n);      % accuracy
sen=(t_p-err_p)/t_p;      % sensitivity
spe=(t_n-err_n)/t_n;      % specificity
summ = alpha_p*sen + alpha_n*spe;  % weighted Sum
fprintf(1,'The sum of weighted sensitivity and specificity = %d\n', summ);  

que = b / B;              % query ratio
classifier.w = w;         % the current predicitive vector
classifier.covariance = covariance;  % the current covariance matrix
run_time = toc;           % the whole time

