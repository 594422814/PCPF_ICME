function x = get_features(im, cell_size, cos_window, w2c)
%GET_FEATURES
%   Extracts dense features from image.
%
%   X = GET_FEATURES(IM, FEATURES, CELL_SIZE)
%   Extracts features specified in struct FEATURES, from image IM. The
%   features should be densely sampled, in cells or intervals of CELL_SIZE.
%   The output has size [height in cells, width in cells, features].
%   To experiment with other features simply add them to this function
%   and include any needed parameters in the FEATURES struct. To allow
%   combinations of features, stack them with x = cat(3, x, new_feat).
%
%   Joao F. Henriques, 2014
%   http://www.isr.uc.pt/~henriques/

    x = double(fhog(single(im) / 255, cell_size));
    x(:,:,end) = [];  %remove all-zeros channel ("truncation feature")
    sz = size(x);    im_patch = imresize(im, [sz(1) sz(2)]);
    out_npca = get_feature_map(im_patch, 'gray', w2c);
    out_pca = get_feature_map(im_patch, 'cn', w2c);
    x = cat(3,x,out_npca);
    % x = cat(3,x,out_pca);   %%%%%%%%%%%%% without CN
    
	%process with cosine window if needed
	if ~isempty(cos_window),
		x = bsxfun(@times, x, cos_window);
	end
	
end
