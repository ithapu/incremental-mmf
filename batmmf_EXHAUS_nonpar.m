
%% ideal MMF code -- batch (No parallel)

function [Us, Wids, Sids, Acomps, H, L, time] = batmmf_EXHAUS_nonpar(A, k, file)
%
tic;
fprintf('\n ========= \n Exhaustive batch Higher-order MMF (No Parallel Operations) \n ========= \n');
%
%% inputs and initializations
if ~exist('A','var') error('\n No inputs!! \n'); end
n = size(A,1); % size of the input matrix
if ~exist('k','var') k = 2; end % # of wavelets stripped at each level
if k>5 
    error('very high-order for the mmf! Computer would crash!! \n'); 
end
if ~exist('L','var') L = n-(k-1); end % # of multi-resolutions spaces (U_1 to U_L)
S = [1:1:n]; % the dimensions to be resolved
%
%% generating the k-point rotation matrices (rand orth matrices)
load(sprintf('OrthMats%d.mat',k)); %% lots of orthogonal matrices
%
%% loop over levels 1 to L
Us = cell(L,1); % computed orth matrices at L levels
Acomps = cell(L,1); % compressed matrices at L levels
Sids = zeros(L,k-1); % indices/rows corresponding to scaling at L levels
Wids = zeros(L,1); % indices/rows corresponding to wavelets at L levels
for l = 1:1:L
    fprintf('\n ------> \t [[ %s ]] \t Level %d/%d \t Order %d \n',file,l,L,k);
    % initialization
    if l>1
        A_lprev = A_l; % current approximation of the input martix
        S_lprev = S_l; % dimensions left out to be resolved
    else
        A_lprev = A; S_lprev = S;
    end
    %
    %% generating the k-tuple indices to compute errors
    % allowed values of k include 2, 3, 4 and 5
    sort_S_lprev = sort(S_lprev,'ascend');
    ls = length(sort_S_lprev); %combs = zeros(ls^k,k);
    if k<=ls
        if k == 2
            [x,y] = ndgrid(1:ls,1:ls); x = x(:); y = y(:);
            combs = [x,y]; clear x y;
        elseif k == 3
            [x,y,z] = ndgrid(1:ls,1:ls,1:ls); x = x(:); y = y(:);
            z = z(:); combs = [x,y,z]; clear x y z;
        elseif k == 4
            [x,y,z,u] = ndgrid(1:ls,1:ls,1:ls,1:ls); x = x(:); y = y(:);
            z = z(:); u = u(:); combs = [x,y,z,u]; clear x y z u;
        elseif k == 5
            [x,y,z,u,v] = ndgrid(1:ls,1:ls,1:ls,1:ls,1:ls);
            x = x(:); y = y(:); z = z(:); u = u(:); v = v(:);
            combs = [x,y,z,u,v]; clear x y z u v;
        else error('\n k too large! reduce it! \n'); end
        if size(combs,1)~=ls^k
            error('\n Something wrong in listing of cominations! \n'); end
    else
        error('\n Inconsistency with # of rows left out! Increase L! \n');
    end
    %% removing irrelevant ones from combs
    combs = sort_S_lprev(combs);
    if k == 2
        remind = (combs(:,1)-combs(:,2)~=0);
    elseif k == 3
        remind = (combs(:,1)-combs(:,2)~=0) + (combs(:,1)-combs(:,3)~=0) ...
            + (combs(:,2)-combs(:,3)~=0);
    elseif k == 4
        remind = (combs(:,1)-combs(:,2)~=0) + (combs(:,1)-combs(:,3)~=0) ...
            + (combs(:,1)-combs(:,4)~=0) + (combs(:,2)-combs(:,3)~=0) ...
            + (combs(:,2)-combs(:,4)~=0) + (combs(:,3)-combs(:,4)~=0);
    elseif k == 5
        remind = (combs(:,1)-combs(:,2)~=0) + (combs(:,1)-combs(:,3)~=0) ...
            + (combs(:,1)-combs(:,4)~=0) + (combs(:,1)-combs(:,5)~=0) ...
            + (combs(:,2)-combs(:,3)~=0) + (combs(:,2)-combs(:,4)~=0) ...
            + (combs(:,2)-combs(:,5)~=0) + (combs(:,3)-combs(:,4)~=0) ...
            + (combs(:,3)-combs(:,5)~=0) + (combs(:,4)-combs(:,5)~=0);
    end  
    removing = setdiff(1:1:size(combs,1),find(remind==k*(k-1)/2)); 
    combs(removing,:) = []; combs = sort(combs,2,'ascend');
    combs = unique(combs,'rows');    
    %
    %% computing the best orth matrices for each of the
    % above combinations via minimizing the errors E_I
    Es_I = zeros(size(combs,1),1); min_Os_Cs_inds = zeros(size(combs,1),1);
    wavepos_inds = zeros(size(combs,1),1);
    %
    for c = 1:1:size(combs,1)
        %
        A1 = A_lprev(combs(c,:),combs(c,:)); 
        B1 = A_lprev(combs(c,:),setdiff(S_lprev,combs(c,:))); B = B1*B1'; 
        term1 = mmat(mmat(Os, repmat(A1, [1 1 searchnum])), Os_tran);
        term2 = mmat(mmat(Os, repmat(B, [1 1 searchnum])), Os_tran);
        %
        term1pool = permute(term1, [2 1 3]) .* repmat(1-eye(k), [1 1 searchnum]);
        term2pool = term2 .* repmat(eye(k), [1 1 searchnum]);
        Es_c = (2*squeeze(sum(term1pool)) + 2*squeeze(sum(term2pool)))';
        [Es_I(c,1),q] = min(reshape(Es_c,1,[])); 
        [min_Os_Cs_inds(c,1), wavepos_inds(c,1)] = ind2sub(size(Es_c),q); 
    end    
    %
    %% set the wavelets and corresponing orth transform U_l
    [~,min_ind] = min(Es_I); U_l = eye(n,n);
    I_l = combs(min_ind,:); O_l = squeeze(Os(:,:,min_Os_Cs_inds(min_ind,1))); 
    for j1 = 1:1:k
        for j2 = 1:1:k
            U_l(I_l(j1),I_l(j2)) = O_l(j1,j2);
        end
    end
    %
    %% setting things for net iteration over l
    A_l = U_l*A_lprev*U_l';
    pickW = wavepos_inds(min_ind,1); % the wavelet
    %
    %% saving things for reference
    Sids(l,:) = setdiff(I_l,I_l(pickW)); % scaling row at each l
    Wids(l,1) = I_l(pickW); % wavelet row at each l
    S_l = setdiff(S_lprev,I_l(pickW));        
    Us{l,1} = U_l; % orth tranform at level l
    Acomps{l,1} = A_l; % compressed mat at level l
    %
end
fprintf('\n===========\n done \n=========\n');
% zeroing out things for final H
H = A_l; dH = diag(H); H_S_L = H(S_l,S_l);
H = zeros(n,n); H(S_l,S_l) = H_S_L;
H = H + diag(dH,0);
%
time = toc;
%
end
