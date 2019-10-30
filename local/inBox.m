
function b = inBox(WinSize, pos, local, rate)

    BoundUR = pos + rate*WinSize/2;
    BoundDL = pos - rate*WinSize/2;

    localUR = local.pos + local.target_sz/2;
    localDL = local.pos - local.target_sz/2;
    
    if min(localUR < BoundUR) * min(localDL > BoundDL) == 0
        b = false;
    else
        b = true;
    end
    
end
