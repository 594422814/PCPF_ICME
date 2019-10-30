function [local] = LocalDCF(im, local, p, global_pos, w2c, target_sz)

if ~isfield(local,'window_sz')
    average_size = sum(local.target_sz(:))/2;
	%window size, taking padding into account
	local.window_sz = floor(local.target_sz + p.padding*average_size);	 
	%create regression labels, gaussian shaped, with a bandwidth
	%proportional to target size
	output_sigma = sqrt(prod(local.target_sz)) * p.output_sigma_factor/p.hog_cell_size;
	local.yf = fft2(gaussian_shaped_labels(output_sigma, floor(local.window_sz/p.hog_cell_size)));
	local.cos_window = hann(size(local.yf,1)) * hann(size(local.yf,2))';	
	%obtain a subwindow for training at newly estimated target position
    patch = get_subwindow(im, local.pos, local.window_sz);
    xf = fft2(get_features(patch, p.hog_cell_size, local.cos_window, w2c));       
    kf = linear_correlation(xf, xf);
    alphaf = local.yf ./ (kf + p.lambda);   %equation for fast training
    local.learning_rate = 0.01;
    local.model_alphaf = alphaf;
    local.model_xf = xf;
    local.psr = 1;
    local.aver_psr = [];
    local.reliability = true;
else
    coarse_pos = global_pos - local.displace;
    %obtain a subwindow for detection at the position from last
    %frame, and convert to Fourier domain (its size is unchanged)
    patch = get_subwindow(im, coarse_pos, local.window_sz);
    zf = fft2(get_features(patch, p.hog_cell_size, local.cos_window, w2c));
    %calculate response of the classifier at all shifts
    kzf = linear_correlation(zf, local.model_xf);
    response = real(ifft2(local.model_alphaf .* kzf));  %equation for fast detection
    %target location is at the maximum response. we must take into
    %account the fact that, if the target doesn't move, the peak
    %will appear at the top-left corner, not at the center (this is
    %discussed in the paper). the responses wrap around cyclically.
    [vert_delta, horiz_delta] = find(response == max(response(:)), 1);
    if vert_delta > size(zf,1) / 2,  %wrap around to negative half-space of vertical axis
        vert_delta = vert_delta - size(zf,1);
    end
    if horiz_delta > size(zf,2) / 2,  %same for horizontal axis
        horiz_delta = horiz_delta - size(zf,2);
    end
    local.pos = coarse_pos + p.hog_cell_size * [vert_delta - 1, horiz_delta - 1];
    local.rect_position = [local.pos([2,1]) - local.target_sz([2,1])/2, local.target_sz([2,1])];
    % motion direction
    new_displace = global_pos - local.pos;
    norm_old = norm(local.displace);
    norm_new = norm(new_displace);
    direction = dot(new_displace, local.displace)/(norm_old*norm_new);
    % ignore the vector less than 5 pixels, whose direction change is hard to estimatre
    motion_angle_false = (min(norm_old,norm_new) > 5)&&(direction < 0);
    motion_range_false = (norm_new > p.safeDistance*sum(target_sz)/2);
    % PSR reliability
    % local.psr = calculatePSR(response);
    local.psr = calculatePSR(response);
    local.aver_psr = [local.aver_psr, local.psr];
    aver_psr = sum(local.aver_psr)/numel(local.aver_psr);
    
    if (local.psr < p.deleteThreshold * aver_psr)||(motion_angle_false == 1)||(motion_range_false == 1)
       local.temporary = local.temporary + 1;
       local.reliability = false;
    else
       local.temporary = 0;
       local.reliability = true;
       %obtain a subwindow for training at newly estimated target position
       patch = get_subwindow(im, local.pos, local.window_sz);
       xf = fft2(get_features(patch, p.hog_cell_size, local.cos_window, w2c));
       %Kernel Ridge Regression, calculate alphas (in Fourier domain)
       kf = linear_correlation(xf, xf);
       alphaf = local.yf ./ (kf + p.lambda);   %equation for fast training
       %subsequent frames, interpolate model
       local.model_alphaf = (1 - local.learning_rate) * local.model_alphaf + local.learning_rate * alphaf;
       local.model_xf = (1 - local.learning_rate) * local.model_xf + local.learning_rate * xf;
    end
    
end
