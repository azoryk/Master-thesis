function output_img = fast_resize(img, m, method)
%by James Hays
%like matlab's resize, except it calls opencv.
%always uses bilinear.  method is completely ignored.
%results don't look as nice as Matlab's for some reason.

%this next bit is taken straight from matlab's imresize
% Define old and new image sizes, and actual scaling
[so(1),so(2),thirdD] = size(img);% old image size
if length(m)==1,% m is a multiplier
    sn = max(floor(m*so(1:2)),1);% new image size=(integer>0)
else            % m is new image size
    sn = m;
end

is_double = 0;
if(isa(img, 'double'))
    img = single(img);
    is_double = 1;
end

if(~exist('method'))
    method = 'bilinear';
end

%bilinear by default. or is it?? apparantly not.
% tic
output_img = cvlib_mex('resize', img, sn, method);
% toc

%to preserve the type of the input_image
if(is_double)
    output_img = double(output_img);
end