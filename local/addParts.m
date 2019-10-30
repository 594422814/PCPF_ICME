
function [ local ] = addParts(particles, pos, target_sz)

n = size(particles, 2);
if n==0
    local = {};
    return
end

sigma = particles([3,4,3,4],:);       
sigma(3:4,:) = repmat(sqrt(sum(sigma(3:4,:).^2,1)), [2,1]);
par = particles + randn(4,n).*sigma;

for i = 1 : n
    local{i} = {};
    p = round(par(1:2,i)');
    local{i}.pos = p;  
    sz = round(par(3:4,i)');
 
    % min local part sz must > HOG 2times cell size (8)
    if min(sz > [target_sz(1), target_sz(2)]) > 0 && min(sz < [target_sz(1), target_sz(2)] ) > 0
        local{i}.target_sz = max(sz, 8); 
    else
        local{i}.target_sz = max(0.4*target_sz, 8);  %%%% 0.2
    end
    
    if ~inBox(target_sz, pos, local{i}, 1) 
        % randnum in range of [-1, 1];
        randnum1 = rand() - rand();    
        randnum2 = rand() - rand();
        local{i}.pos = pos + [randnum1, randnum2].*target_sz/4;
    end
    local{i}.temporary = 0;
    local{i}.displace = pos - p;
    local{i}.rect_position = [local{i}.pos([2,1]) - local{i}.target_sz([2,1])/2, local{i}.target_sz([2,1])];
end

end
