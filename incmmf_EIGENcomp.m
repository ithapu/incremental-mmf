
%% eigen MMF code -- ONLINE

function [Us_o, Wids_o, Sids_o, Acomps_o, H_o, L_o, time_o] = incmmf_EIGENcomp(C, k, complete, comp_opt, lossy_or_lossless, file)
%
tic;
fprintf('\n ========= \n Incremental Higher-order MMF \n');
fprintf(' (Completion is Eigen-based; Initialization is user-defined) \n ========= \n');
%
if length(complete)<=k 
    error('The order of MMF should be < than #complete elements! \n'); end
% the sub-matrix for complete mmf
if strcmp(comp_opt,'exhaustive')
    if length(complete)>15
        fprintf('Too many complete entries to perform ideal MMF initialization!! \n');
        fprintf('\t Switching instead to random based one \n');
        comp_opt = 'random';
    else
        [Us_0, Wids_0, Sids_0, Acomps_0, H_0, L_0] = batmmf_EXHAUS_nonpar(C(complete, complete), k, file);
    end
end
if strcmp(comp_opt,'random')
    [Us_0, Wids_0, Sids_0, Acomps_0, H_0, L_0] = batmmf_RANDOM(C(complete, complete), k, file);
elseif strcmp(comp_opt,'eigen')
    [Us_0, Wids_0, Sids_0, Acomps_0, H_0, L_0] = batmmf_EIGEN_nonpar(C(complete, complete), k, file);
end
% the rest
rest = setdiff(1:1:size(C,1),complete);

%% going thru ol factorization for all the rest indices
%
for r = 1:1:length(rest)
    %
    fprintf('\n ------> \t [[ %s ]] \t Filling element #%d/%d \t Order %d \t (%s, Complete %s) \n',...
        file, r, length(rest), k, lossy_or_lossless, comp_opt);
    %% current filling in
    A = C(complete, complete);
    n = size(A,1); UV = C(rest(r),union(complete,rest(r)));
    SW_stack = [Sids_0 Wids_0]; clear Sids_0 Wids_0;
    if size(UV,1)~=1 UV = UV'; end
    L = L_0 + 1; % new number of levels
    %
    %% initializations for lossless settnigs
    Us = cell(L,1); % computed orth matrices at L levels
    Acomps = cell(L,1); % compressed matrices at L levels
    Sids = zeros(L,k-1); % indices/rows corresponding to scales at L levels
    Wids = zeros(L,1); % indices/rows corresponding to wavelets at L levels
    insert = zeros(1,L_0+1); insert(end) = 1; % insertion points
    % the big matrix
    Abig = [A; UV(1:end-1)]; Abig = [Abig, UV'];
    S = [1:1:n+1]; % the dimensions to be resolved including the new one
    %
    %% size adjusting for the rotation matrices;
    for l = 1:1:numel(Us_0)
        U_l = Us_0{l,1}; U_l = [U_l; zeros(1,n)];
        U_l = [U_l, [zeros(n,1); 1]]; Us_0{l,1} = U_l;
    end
    clear U_l;
    %
    %% find the insertions and updating the wavelet tree
    %
    id_update = [1+n]; % the extra insertion vector and index
    % finding isertions across all the L-1 levels
    %
    for l = 1:1:L
        %
        fprintf('\t \t \t Insertion Level %d/%d \n',l,L);
        %
        if isempty(id_update) & l<L
        error('No insertions required while tree leaves are not reached! Something wrong!! \n'); end
        %
        % the compression at previous level used to calculate the rotations
        % the available rows at the current level
        if l>1 A_lprev = Acomps{l-1,1}; S_lprev = S_l;
        else A_lprev = Abig; S_lprev = S; end
        % selecting the current best choice of rows from batch mmf
        if l<L
            % we are at l<L and hence stack of SWs is not empty yet
            combs = SW_stack(l,:); % first k-1 are scaling, last is wavelet        
        else combs = []; end
        %
        %% old error
        % checking if the old combs are valid
        if l<L
            if ~isempty(intersect(combs,Wids(1:l-1,1)))
                err_old = inf;
            else
                % if valid then compute the error
                B1 = A_lprev(combs,setdiff(S_lprev,combs)); B = B1*B1';
                Ocurr = Us_0{l,1}(combs,combs);
                term1 = Ocurr*A_lprev(combs,combs)*Ocurr'; term2 = Ocurr*B*Ocurr';
                term1pool = sum((term1*term1').*(1-eye(k)),2);
                err_old = 2*(term1pool(k) + term2(k,k));
            end
        else err_old = inf; end
        %
        %% new set of errors (k different new combs) -- inserting each of the id_update entries
        % the new mega combs set
        if l>1 & l<L 
            thelist = setdiff(union(combs,id_update),Wids(1:l-1,1));
        elseif l==1
            thelist = union(combs,id_update);
        else
            thelist = setdiff(1:1:n+1,Wids(1:l-1,1));
        end
        %
        % first listng out possible new combs and then computing their errors
        total_combs = combnk(thelist,k); num_total_combs = size(total_combs,1);
        newcombs_errs = zeros(1,num_total_combs); 
        new_Orth = cell(size(num_total_combs,1),1);
        new_wave_pos_list = zeros(1,num_total_combs);
        %
        % computing the errors
        for idlen = 1:1:num_total_combs
            newcombs = total_combs(idlen,:);
            A1 = A_lprev(newcombs,newcombs);
            B1 = A_lprev(newcombs,setdiff(S_lprev,newcombs)); B = B1*B1';
            % computing O
            [V, D] = eig(A1*A1'); term1 = V*A1*V'; term2 = V*B*V'; 
            term1pool = sum((term1*term1').*(1-eye(k)),2);
            Es_c = 2*(term1pool + diag(term2));
            [newcombs_errs(1,idlen), new_wave_pos_list(1,idlen)] = min(Es_c);
            new_Orth{idlen,1} = V;
        end
        %
        %% choosing the best knocking index
        [err_new, best_comb_idlen] = min(newcombs_errs);
        %
        if err_new >= err_old
            % nothing to insert -- move on to next level
            insert(1,l) = -1;
            % the old orth transform adjusted to new dimensions
            if strcmp(lossy_or_lossless,'lossy')
                U_l = Us_0{l};
            else
                A1 = A_lprev(combs,combs); 
                B1 = A_lprev(combs,setdiff(S_lprev,combs)); B = B1*B1'; 
                % computing O
                [V, D] = eig(A1*A1');
                O_l = V; U_l = eye(n+1,n+1);
                for j1 = 1:1:k
                    for j2 = 1:1:k
                        U_l(combs(j1),combs(j2)) = O_l(j1,j2);
                    end
                end
            end
        else
            % there has been an insertion -- check it
            insert(1,l) = 1;
            new_wave_pos = new_wave_pos_list(1, best_comb_idlen);
            %
            % the new combination
            newcombs = zeros(1,k); newcombs(1,end) = total_combs(best_comb_idlen,new_wave_pos);
            newcombs(1:end-1) = total_combs(best_comb_idlen,setdiff(1:1:k,new_wave_pos));
            % update the rotation
            O_l = new_Orth{best_comb_idlen,1}; U_l = eye(n+1,n+1);
            for j1 = 1:1:k
                for j2 = 1:1:k
                    U_l(newcombs(j1),newcombs(j2)) = O_l(j1,j2);
                end
            end
            %
            % the inserting elements for next level
            id_update_old = id_update; clear id_update;
            if l<L
                if combs(end)~=newcombs(end) id_update = combs(end); 
                else id_update = []; end
                for p = 1:1:length(id_update_old)
                    if id_update_old(p)~=newcombs(end) id_update = [id_update, id_update_old(p)]; end
                end
            else id_update = []; end
        end
        %
        %% apply the rotation
        A_l = U_l*A_lprev*U_l';
        % saving and updating things for the next step
        Acomps{l,1} = A_l; Us{l,1} = U_l; % saving things
        % saving the wavelet and scaling indices
        if insert(1,l)==-1 
            Wids(l,1) = combs(end); Sids(l,:) = combs(1:end-1);
        else
            Wids(l,1) = newcombs(end); Sids(l,:) = newcombs(1:end-1);
        end
        % the rest of the rows/columns that need to be resolved
        S_l = setdiff(S_lprev,Wids(l,1)); % updating things        
        %
    end
    %
    %% zeroing out things for final H
    H = A_l; dH = diag(H); H_S_L = H(S_l,S_l); H = zeros(n+1,n+1); H(S_l,S_l) = H_S_L;
    H = H + diag(dH,0);
    %
    %% setting things for next rest index
    Us_0 = Us; Wids_0 = Wids; Sids_0 = Sids; Acomps_0 = Acomps;
    H_0 = H; L_0 = L; clear Us Wids Sids Acomps H L; complete = union(complete,rest(r));
    
end    
    
%% some more checking and final saving
if ~isempty(setdiff(complete,1:1:size(C)))
    error('\n All rows/cols not done! \n'); end
% all done
fprintf('\n======================\n done \n=====================\n');
Us_o = Us_0; Wids_o = Wids_0; Sids_o = Sids_0; Acomps_o = Acomps_0;
H_o = H_0; L_o = L_0; clear Us_0 Wids_0 Sids_0 Acomps_0 H_0 L_0;
%
time_o = toc;
%
end

%%
