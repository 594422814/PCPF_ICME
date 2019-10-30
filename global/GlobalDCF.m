function global_pos = GlobalDCF(im_patch_cf, p ,hann_window, indLayers, area_resize_factor, pos, hf_num_deep, hf_den_deep )

xt_deep  = getDeepFeatureMap(im_patch_cf, hann_window, indLayers);

for ii = 1 : length(indLayers)
   xtf_deep = fft2(xt_deep{ii});
   hf_deep = bsxfun(@rdivide, hf_num_deep{ii}, sum(hf_den_deep{ii}, 3) + p.lambda);                     
   response_deep{ii} = ensure_real( ifft2(sum( conj(hf_deep) .* xtf_deep, 3))  );
end

for ii = 1 : length(indLayers)
  responseDeep{ii} = cropFilterResponse(response_deep{ii}, floor_odd(p.norm_delta_area / p.hog_cell_size));
  responseDeep{ii} = mexResize(responseDeep{ii}, p.norm_delta_area, 'auto');
end

response = responseDeep{1} + 0.5*responseDeep{2} + 0.02*responseDeep{3};
center = (1 + p.norm_delta_area)/2;    
[row, col] = find(response == max(response(:)), 1);    
global_pos = pos + ([row, col] - center) / area_resize_factor;   
      
end

% We want odd regions so that the central pixel can be exact
function y = floor_odd(x)
    y = 2*floor((x-1) / 2) + 1;
end
function y = ensure_real(x)
    assert(norm(imag(x(:))) <= 1e-5 * norm(real(x(:))));
    y = real(x);
end
