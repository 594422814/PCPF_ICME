%% calculatePSR

function PSR = calculatePSR(response)
%    % PSR
    maxvalue = max(response(:));
    average = sum(response(:))/(size(response,1)*size(response,2));
    sigma = sqrt(sum(sum((response - average).^2))/(size(response,1)*size(response,2))); 
    PSR = (maxvalue - average)/sigma;

% % APCE
%      maxvalue = max(response(:));
%      minvalue = min(response(:));
%      square_matrix = (response - minvalue).^2;
%      PSR = (maxvalue - minvalue).^2/mean(square_matrix(:));
end