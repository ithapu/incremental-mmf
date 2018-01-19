
function visualize_mmf(Wids_o, Sids_o, names)

todispbag = []; fprintf('\n');
%
for i = 1:1:length(Wids_o)
    %
%     fprintf('\t');
    for j = 1:1:length(Sids_o(i,:))
        if isempty(find(todispbag==Sids_o(i,j)))
            fprintf('%s\t',names{Sids_o(i,j)});
            todispbag = [todispbag, Sids_o(i,j)];
        else
            fprintf('-c-\t');
        end
    end
    fprintf('|');
    if isempty(find(todispbag==Wids_o(i)))
        fprintf('%s\t',names{Wids_o(i)});
        todispbag = [todispbag, Wids_o(i)];
    else
        fprintf('-c-\t');
    end
    %fprintf('\t%s',names{Wids_o(i)});
    fprintf('\n\n');
end
%
fprintf('\n');

end