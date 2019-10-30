clc;
clear all;
% RUN_TRACKER  is the external function of the tracker - does initialization and calls trackerMain
%% path
addpath('./global');                        % from Staple
addpath('./local');
addpath('./utility');
addpath('model','external/matconvnet/matlab');
vl_setupnn();

%% load video info
videoname = 'Bird2'; 
base_path = 'sequence/';
img_path = 'sequence/Bird2/img/';
[img_files, pos, target_sz, video_path] = load_video_info(base_path, videoname);

%% debugging stuff 
params.visualization = 1;                 % show output bbox of global correlation filter
params.visualization_dbg = 1;             % show output bboxes of local correlation particle filters
%% tracking-by-detection
params.hog_cell_size = 4;
params.fixed_area = 200^2;                 % standard area to which we resize the target, use 150^2 to get higher speed 
params.n_bins = 2^5;                       % number of bins for the color histograms (bg and fg models)
params.learning_rate_pwp = 0.01;           % bg and fg color models learning rate 
params.inner_padding = 0.2;                % defines inner area used to sample colors from the foreground
params.output_sigma_factor = 0.1;          % standard deviation for the desired translation filter output
params.lambda = 1e-4;                      % regularization weight
params.learning_rate_cf = 0.01;            % learning rate
params.localPars = 100;                    % particle number
params.deleteThreshold = 0.7;              % 0.7
params.safeDistance = 1.2;
%% scale related
params.hog_scale_cell_size = 4;            % from DSST 
params.learning_rate_scale = 0.025;
params.scale_sigma_factor = 1/2;
params.num_scales = 33;
params.scale_model_factor = 1.0;
params.scale_step = 1.03;
params.scale_model_max_area = 32*16;
%% start trackerMain.m
im = imread([img_path img_files{1}]);
% is a grayscale sequence ?
if(size(im,3)==1)
    params.grayscale_sequence = true;
else
    params.grayscale_sequence = false;
end
params.img_files = img_files;
params.img_path = img_path;
% init_pos is the centre of the initial bounding box
params.init_pos = pos;
params.target_sz = target_sz;
[params, bg_area, fg_area, area_resize_factor] = initializeAllAreas(im, params);
if params.visualization
    params.videoPlayer = vision.VideoPlayer('Position', [100 100 [size(im,2), size(im,1)]+30]);
end
% start the actual tracking
trackerMain(params, im, bg_area, fg_area, area_resize_factor);

