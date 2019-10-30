function [results] = trackerMain(p, im, bg_area, fg_area, area_resize_factor)

temp = load('w2crs');
w2c = temp.w2crs;
pos = p.init_pos;
target_sz = p.target_sz;
num_frames = numel(p.img_files);

% patch of the target + padding
patch_padded = getSubwindow(im, pos, p.norm_bg_area, bg_area);
% initialize hist model
new_pwp_model = true;
[bg_hist, fg_hist] = updateHistModel(new_pwp_model, patch_padded, bg_area, fg_area, target_sz, p.norm_bg_area, p.n_bins, p.grayscale_sequence);
new_pwp_model = false;

% Hann (cosine) window
hann_window_cosine = single(hann(p.cf_response_size(1)) * hann(p.cf_response_size(2))');
% gaussian-shaped desired response, centred in (1,1)
% bandwidth proportional to target size
output_sigma = sqrt(prod(p.norm_target_sz)) * p.output_sigma_factor / p.hog_cell_size;
y = gaussianResponse(p.cf_response_size, output_sigma);
yf = fft2(y);
%% SCALE ADAPTATION INITIALIZATION
% Code from DSST
scale_factor = 1;
base_target_sz = target_sz;
scale_sigma = sqrt(p.num_scales) * p.scale_sigma_factor;
ss = (1:p.num_scales) - ceil(p.num_scales/2);
ys = exp(-0.5 * (ss.^2) / scale_sigma^2);
ysf = single(fft(ys));
if mod(p.num_scales,2) == 0
    scale_window = single(hann(p.num_scales+1));
    scale_window = scale_window(2:end);
else
    scale_window = single(hann(p.num_scales));
end;
ss = 1:p.num_scales;
scale_factors = p.scale_step.^(ceil(p.num_scales/2) - ss);
if p.scale_model_factor^2 * prod(p.norm_target_sz) > p.scale_model_max_area
    p.scale_model_factor = sqrt(p.scale_model_max_area/prod(p.norm_target_sz));
end
scale_model_sz = floor(p.norm_target_sz * p.scale_model_factor);
% find maximum and minimum scales
min_scale_factor = p.scale_step ^ ceil(log(max(5 ./ bg_area)) / log(p.scale_step));
max_scale_factor = p.scale_step ^ floor(log(min([size(im,1) size(im,2)] ./ target_sz)) / log(p.scale_step));

indLayers = [37, 28, 19];   % The CNN layers Conv5-4, Conv4-4, Conv3-4 in VGG Net
numLayers = length(indLayers);
new_hf_num_deep = cell(1,2);        hf_den_deep = cell(1,2);
new_hf_den_deep = cell(1,2);        hf_num_deep = cell(1,2);
xtf_deep = cell(1,2);

% Main Loop
for frame = 1 : num_frames
           
if frame>1
      frame
     %% Global DCF
      im = imread([p.img_path p.img_files{frame}]); 
      % extract patch of size bg_area and resize to norm_bg_area
      im_patch_cf = getSubwindow(im, pos, p.norm_bg_area, bg_area);
      % color histogram
      [likelihood_map] = getColourMap(im_patch_cf, bg_hist, fg_hist, p.n_bins, p.grayscale_sequence);
      likelihood_map(isnan(likelihood_map)) = 0;
      likelihood_map = imResample(likelihood_map, p.cf_response_size);
      likelihood_map = (likelihood_map + min(likelihood_map(:)))/(max(likelihood_map(:)) + min(likelihood_map(:)));  
      likelihood_map(isnan(likelihood_map)) = 0;
      if ( (sum(likelihood_map(:))/prod(p.cf_response_size))<0.01 ) 
         likelihood_map = 1; 
      else
         likelihood_map = max(likelihood_map, 0.1);  % To ensure not too many zero
      end    
      hann_window =  hann_window_cosine.* likelihood_map;   
      global_pos = GlobalDCF(im_patch_cf, p ,hann_window, indLayers, area_resize_factor, pos, hf_num_deep, hf_den_deep);

     %% local DCF, track the target parts
       for i = 1:numel(local)
          local{i} = LocalDCF(im, local{i}, p, global_pos, w2c, target_sz);
       end
      [local_pos, false_num] = voting(local);
      pos = local_pos;

      false_ratio = false_num / p.localPars; 
      if(false_ratio > 0.8),  pos = global_pos;   end
      local = resetParts(pos, target_sz, local, p, false_num);
      for i = 1 : numel(local)
          if local{i}.reliability == true   % reliable particles
             local{i}.displace = pos - local{i}.pos;
          end
      end

      p.learning_rate_cf = 0.01 * exp(-1*false_ratio^2/0.01);  
      p.learning_rate_pwp = 0.01;
      if(false_ratio > 0.2),  p.learning_rate_pwp = 0;   end
    
      global_rect = [global_pos([2,1]) - target_sz([2,1])/2, target_sz([2,1])];
      rect_position = [pos([2,1]) - target_sz([2,1])/2, target_sz([2,1])]; 
    
     %% SCALE SPACE SEARCH
      im_patch_scale = getScaleSubwindow(im, pos, base_target_sz, scale_factor * scale_factors, scale_window, scale_model_sz, p.hog_scale_cell_size);
      xsf = fft(im_patch_scale,[],2);
      scale_response = real(ifft(sum(sf_num .* xsf, 1) ./ (sf_den + p.lambda) ));
      recovered_scale = ind2sub(size(scale_response),find(scale_response == max(scale_response(:)), 1));
      % set the scale
      scale_factor = scale_factor * scale_factors(recovered_scale);
      if scale_factor < min_scale_factor
          scale_factor = min_scale_factor;
      elseif scale_factor > max_scale_factor
          scale_factor = max_scale_factor;
      end
      % use new scale to update bboxes for target, filter, bg and fg models
      target_sz = round(base_target_sz * scale_factor);
      p.avg_dim = sum(target_sz)/2;
      bg_area = round(target_sz + p.avg_dim * p.padding);
      if(bg_area(2)>size(im,2)),  bg_area(2)=size(im,2)-1;    end
      if(bg_area(1)>size(im,1)),  bg_area(1)=size(im,1)-1;    end

      bg_area = bg_area - mod(bg_area - target_sz, 2);
      fg_area = round(target_sz - p.avg_dim * p.inner_padding);
      fg_area = fg_area + mod(bg_area - fg_area, 2);
      % Compute the rectangle with (or close to) params.fixed_area and same aspect ratio as the target bboxgetScaleSubwindow
      area_resize_factor = sqrt(p.fixed_area/prod(bg_area));
end

  %% TRAINING
    % extract patch of size bg_area and resize to norm_bg_area
    im_patch_bg = getSubwindow(im, pos, p.norm_bg_area, bg_area);
    xt_deep  = getDeepFeatureMap(im_patch_bg, hann_window_cosine, indLayers);
    for ii = 1 : numLayers
       xtf_deep{ii} = fft2(xt_deep{ii});
       new_hf_num_deep{ii} = bsxfun(@times, conj(yf), xtf_deep{ii}) / prod(p.cf_response_size);
       new_hf_den_deep{ii} = (conj(xtf_deep{ii}) .* xtf_deep{ii}) / prod(p.cf_response_size);
    end
    
    if frame == 1
       % local DCF tracker
       pars = repmat([pos target_sz * 0.4]',[1 p.localPars]);   %%%%%%%%%%%%%%%%%% 0.2
       local = addParts(pars, pos, target_sz);
       %track the parts
       for i = 1 : numel(local)
           local{i} = LocalDCF(im, local{i}, p, pos, w2c, target_sz);
       end 
       % global tracker. 
       for ii = 1 : numLayers 
          hf_den_deep{ii} = new_hf_den_deep{ii};
          hf_num_deep{ii} = new_hf_num_deep{ii};
       end
    else
       % subsequent frames, update the model by linear interpolation
       for ii= 1 : numLayers
          hf_den_deep{ii} = (1 - p.learning_rate_cf) * hf_den_deep{ii} + p.learning_rate_cf * new_hf_den_deep{ii};
          hf_num_deep{ii} = (1 - p.learning_rate_cf) * hf_num_deep{ii} + p.learning_rate_cf * new_hf_num_deep{ii};
       end
       % BG/FG MODEL UPDATE   patch of the target + padding
       im_patch_color = getSubwindow(im, pos, p.norm_bg_area, bg_area*0.8);
       [bg_hist, fg_hist] = updateHistModel(new_pwp_model, im_patch_color, bg_area, fg_area, target_sz, p.norm_bg_area, p.n_bins, p.grayscale_sequence, bg_hist, fg_hist, p.learning_rate_pwp);
    end
    
   %% SCALE UPDATE
    im_patch_scale = getScaleSubwindow(im, pos, base_target_sz, scale_factor*scale_factors, scale_window, scale_model_sz, p.hog_scale_cell_size);
    xsf = fft(im_patch_scale,[],2);
    new_sf_num = bsxfun(@times, ysf, conj(xsf));
    new_sf_den = sum(xsf .* conj(xsf), 1);
    if frame == 1,
        sf_den = new_sf_den;
        sf_num = new_sf_num;
    else
        sf_den = (1 - p.learning_rate_scale) * sf_den + p.learning_rate_scale * new_sf_den;
        sf_num = (1 - p.learning_rate_scale) * sf_num + p.learning_rate_scale * new_sf_num;
    end
    % update bbox position
    if frame == 1, 
        rect_position = [pos([2,1]) - target_sz([2,1])/2, target_sz([2,1])]; 
        global_rect = [pos([2,1]) - target_sz([2,1])/2, target_sz([2,1])];
    end 
    rect_position_padded = [pos([2,1]) - bg_area([2,1])/2, bg_area([2,1])];  
     
   %% VISUALIZATION
    if p.visualization == 1
        if isToolboxAvailable('Computer Vision System Toolbox')
            % im = insertShape(im, 'Rectangle', global_rect, 'LineWidth', 2, 'Color', 'red');
            im = insertShape(im, 'Rectangle', rect_position, 'LineWidth', 4, 'Color', 'red'); %%%%%%%%
            if p.visualization_dbg==1
               for i = 1 : numel(local)
                   if local{i}.reliability == true
                      im = insertShape(im, 'Rectangle', local{i}.rect_position, 'LineWidth', 1, 'Color', 'yellow');
                   else
                      im = insertShape(im, 'Rectangle', local{i}.rect_position, 'LineWidth', 1, 'Color', 'black');
                   end
               end  
            end
            % im = insertShape(im, 'Rectangle', rect_position_padded, 'LineWidth', 2, 'Color', 'yellow');
            % Display the annotated video frame using the video player object.
            step(p.videoPlayer, im);
       else
            figure(1)
            imshow(im)
            rectangle('Position',rect_position, 'LineWidth',2, 'EdgeColor','r');
            rectangle('Position',rect_position_padded, 'LineWidth',2, 'LineStyle','-', 'EdgeColor','y');
            drawnow
       end
    end
    
end

end