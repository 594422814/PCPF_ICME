
function  [position, false_num] = voting(local) 

n = numel(local);
trajec = [];
weight = [];
false_num = 0;

for i = 1:n     
    if local{i}.reliability == false
       false_num = false_num + 1; 
    end
end
disp( [num2str(false_num),' unreliable local particles.']);

for i = 1:n
    if  (local{i}.psr >0) && (local{i}.reliability == true)
        local_pos = local{i}.pos + local{i}.displace;
        trajec = [trajec; local_pos];
        weight = [weight local{i}.psr];
    end
end

weight = weight./sum(weight);
position = weight * trajec;

end

