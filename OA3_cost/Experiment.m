function [m_idx,mPA1, mPA2] = Experiment(dataset_name)  % run Experiment("german")
t1 = clock; 
load(sprintf('data/%s',dataset_name));   

[n,d]       = size(data);    
options.n   = n;             
options.d   = d -1;            
Num_p=sum(data(:,1)==+1);     
Num_n=sum(data(:,1)==-1);    
m           = n;             
options.cost_p = 0.9;      % positive tradeoff for weighted cost
options.cost_n = 0.1;      % negative tradeoff for weighted cost
options.delta = 1;  
options.gamma = 1; 

options.tick=round(n/15);         
options.tick_p=round(Num_p/15);  
options.tick_n=round(Num_n/15);   
options.rho=options.cost_p/options.cost_n;  

options.eta =0.1;     % learning rate for OA3: see the paper for recommended values
options.cost = [0,1; options.rho,0];   
options.B = ceil(n/2);
options.delta_p = 1000;      % positive query parameters 
options.delta_n = 10;        % negative query parameters 

ID_list = ID_ALL;     % the id of 20 random permutation
Y = data(1:m,1);      % label
X = data(1:m,2:d);    % data
X2 = X./repmat(sqrt(sum(X.*X,2)),1, size(X,2));   
X = X2;
ID = ID_list(1,:);

[classifier, cost, sen, spe, acc, que, run_time, TMs, P,idx] = OA3_diag(Y,X,options,ID);
idx = idx; % Record overall idx

parfor i=1:20,
    fprintf(1,'running on the %d-th trial...\n',i);
    ID = ID_list(i,:);
    
   [classifier, cost, sen, spe, acc, que, run_time, TMs, P,idx] = OA3(Y, X, options, ID);
    acc_OA3(i) = acc*100;
    time_OA3(i) = run_time;
    P_list_OA3(i,:) = P;
	sen_OA3(i)=sen*100;
	spe_OA3(i)=spe*100;
    cost_OA3(i)=cost;   
    que_OA3(i) = que*100;
    
    [classifier, cost, sen, spe, acc, que, run_time, TMs, P,idx] = OA3_diag(Y, X, options, ID);
    acc_OA3_diag(i) = acc*100;
    time_OA3_diag(i) = run_time;
    P_list_OA3_diag(i,:) = P;
	sen_OA3_diag(i)=sen*100;
	spe_OA3_diag(i)=spe*100;
    cost_OA3_diag(i)=cost;   
    que_OA3_diag(i) = que*100;
end


figure   
figure_FontSize=17;
mean_P_OA3= mean(P_list_OA3);
plot(idx, mean_P_OA3,'r-o'); 
hold on
mean_P_OA3_diag= mean(P_list_OA3_diag);
plot(idx, mean_P_OA3_diag,'r-+'); 

legend("OA3","OA3-diag");
xlabel('Number of samples');
ylabel('Online average of the sum');
set(get(gca,'XLabel'),'FontSize',figure_FontSize,'Vertical','top');     
set(get(gca,'YLabel'),'FontSize',figure_FontSize,'Vertical','middle');    
set(findobj(get(gca,'Children'),'LineWidth',0.5),'LineWidth',3);  
grid 

fprintf(1,'-------------------------------------------------------------------------------\n');
fprintf('Algorithm : (cost ,   sensitivity,      specificity,     cpu running time)\n');
fprintf('OA3  &%.3f \t$\\pm$ %.3f \t& %.3f \t$\\pm$ %.3f \t& %.3f \t$\\pm$ %.3f \t& %.3f \t\\\\\n', mean(cost_OA3), std(cost_OA3), mean(sen_OA3), std(sen_OA3),mean(spe_OA3), std(spe_OA3), mean(time_OA3));
fprintf('OA3_diag  &%.3f \t$\\pm$ %.3f \t& %.3f \t$\\pm$ %.3f \t& %.3f \t$\\pm$ %.3f \t& %.3f \t\\\\\n', mean(cost_OA3_diag), std(cost_OA3_diag), mean(sen_OA3_diag), std(sen_OA3_diag),mean(spe_OA3_diag), std(spe_OA3_diag), mean(time_OA3_diag));
fprintf(1,'-------------------------------------------------------------------------------\n');
t2 = clock; 
%fprintf('End time$ %.1f \t \\\\\n', t2);
fprintf('Lasting time$ %.3f \t \\\\\n', etime(t2,t1));



