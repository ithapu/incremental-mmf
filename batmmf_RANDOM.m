
%% Non-ideal MMF code -- batch

function [Us, Wids, Sids, Acomps, H, L, time] = batmmf_RANDOM(A, k, file)
%
tic;
fprintf('\n ========= \n Random batch Higher-order MMF \n ========= \n');
%
%% inputs and initializations
if ~exist('A','var') error('\n No inputs!! \n'); end
n = size(A,1); % size of the input matrix
if ~exist('k','var') k = 2; end % # of wavelets stripped at each level
% if k>5 
%     error('very high-order for the mmf! Computer would crash!! \n'); 
% end
if ~exist('L','var') L = n-(k-1); end % # of multi-resolutions spaces (U_1 to U_L)
S = [1:1:n]; % the dimensions to be resolved
%
%% generating the k-point rotation matrices (rand orth matrices)
load(sprintf('OrthMats%d.mat',k)); %% lots of orthogonal matrices
%
%% loop over levels 1 to L
Us = cell(L,1); % computed orth matrices at L levels
Acomps = cell(L,1); % compressed matrices at L levels
Sids = zeros(L,k-1); % indices/rows corresponding to scales at L levels
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
    %
    rp = randperm(length(S_lprev)); 
    % selecting the first scaling index at random
    combs = zeros(1,k); combs(1) = S_lprev(rp(1));
    % selecting the rest of the indices involved in the rotation
    checkrest = setdiff(S_lprev,combs(1));
    nips_checkrest = zeros(1,length(checkrest));
    for i = 1:1:length(checkrest)
        nips_checkrest(1,i) = A_lprev(combs(1),:)*A_lprev(checkrest(i),:)';
        nips_checkrest(1,i) = nips_checkrest(1,i)/norm(A_lprev(checkrest(i),:));
    end
    [~,sort_nips] = sort(nips_checkrest,'descend');
    combs(1,2:end) = checkrest(sort_nips(1:k-1));
    
    %% computing the best orth matrices for each of the
    % above combinations via minimizing the errors E_I
    %Es_I = zeros(1,k); min_Os_Cs_inds = zeros(1,k);
    %
    A1 = A_lprev(combs,combs);
    B1 = A_lprev(combs,setdiff(S_lprev,combs)); B = B1*B1';
    term1 = mmat(mmat(Os, repmat(A1, [1 1 searchnum])), Os_tran);
    term2 = mmat(mmat(Os, repmat(B, [1 1 searchnum])), Os_tran);
    %
    term1pool = permute(term1, [2 1 3]) .* repmat(1-eye(k), [1 1 searchnum]);
    term2pool = term2 .* repmat(eye(k), [1 1 searchnum]);
    Es_c = (2*squeeze(sum(term1pool)) + 2*squeeze(sum(term2pool)))';
    [~,q] = min(reshape(Es_c,1,[]));
    [min_Os_ind, wavepos_ind] = ind2sub(size(Es_c),q);
    %
    %% computing the corresponing orth transform U_l
    I_l = combs; clear combs; O_l = squeeze(Os(:,:,min_Os_ind)); U_l = eye(n,n);
    for j1 = 1:1:k
        for j2 = 1:1:k
            U_l(I_l(j1),I_l(j2)) = O_l(j1,j2);
        end
    end
    %
    %% setting things for next iteration over l
    A_l = U_l*A_lprev*U_l';
    pickW = wavepos_ind; % the wavelet
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
