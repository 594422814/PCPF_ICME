
function local = resetParts(pos, target_sz, local, p, false_num)
%delete the unstable patches

n = numel(local);
index = [];

for i = 1:n
     delete =  ~inBox(target_sz, pos, local{i}, p.safeDistance) || (local{i}.temporary > 4);
     if( delete == 1 )&&( false_num < 0.2 * p.localPars )
        index = [index i];
     end
end

% reset the unreliable particles
if ~isempty(index)
    k = size(index,2);
    pars = repmat([pos target_sz*0.4]',[1 k]);    %%%%%% 0.2 * target_sz %%%%%%%%%%%%%%%%%%%%%
    addPatches = addParts(pars, pos, target_sz);
    for i = 1:size(index,2)
      local{index(i)} = addPatches{i};
      local{index(i)}.reliability = true;
    end   
end

end
