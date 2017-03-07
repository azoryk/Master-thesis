function downloadphotos_int()
%originally written by Tamara Berg, extended by James Hays

%download images listed in every text file in this directory
% search_result_dir = '/nfs/hn22/jhhays/flickr_scripts/search_results/'
search_result_dir = '/nfs/hn26/jhhays/download_scripts/search_results_interesting/'
%directory where you want to download the images
output_dir = ['/nfs/baikal_scratch1/jhhays/flickr_interesting/'];
%the algorithm will create subdirs and subsubdirs in the above dir.

%with the number of images we're expecting we will need two levels of
%subdirectories.  The first level will be the tag, of which there will be
%roughly 100, then the second level will be subdirs of 1000 images.  The
%images will be named such that they can be traced back to their flickr
%source.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('Reading image metadata from %s\n', search_result_dir);
search_results = dir(fullfile(search_result_dir, '*.txt'));
num_results = length(search_results);
rand('twister', sum(100*clock)); %seed random number generator
search_results = search_results(randperm(num_results)); %randomizing order
search_results = [search_results; search_results]; %two passes through to make sure
fprintf(' Downloading the images in %d search results\n',num_results)

for i = 1:size(search_results,1)
    current_filename = search_results(i).name;
    current_filename_fh = current_filename(1:end-4); %cutting off .txt extension 
    
    fprintf('\n !!! Checking for lock on %s\n', current_filename)
    
    %The presence of an output directory is a lock
    if ( ~exist([output_dir current_filename_fh], 'file'))
        
        fprintf(' locking %s\n', current_filename)
        %lock the file by creating the output directory
        cmd = ['mkdir ' output_dir current_filename_fh];
        unix(cmd);
        
        % file where the metadata is located
        fid = fopen([search_result_dir current_filename],'r');
        fprintf(' Reading search results in file %s\n', current_filename);

        count=0;
        dircount=0;

        % downloads images, 1000 images per directory
        while 1
          line = fgetl(fid);
          if (~ischar(line) || count > 2)
            break
          end
          
            %example entry
            % photo: 27642011 8fa8ae33cd 23
            % owner: 72399739@N00
            % title: Mexico City - Chapultepec 03
            % originalsecret: null
            % originalformat: null
            % datetaken: 2005-07-21 14:41:58
            % tags: architecture digital landscape mexico mexicocity infrared trips skyview chapultepec
            % license: 0
            % latitude: 19.420934
            % longitude: -99.181416
            % accuracy: 16
            % interestingness: 2 out of 14
          
            %odd, it seems like original secret is always the same as the
            %standard secret.  And sometimes, if flickr won't give you the
            %original secret, you can still access the original.  So really
            %no point even looking for that.
            
          if strncmp(line,'photo:',6)
              
            if mod(count,1000)==0
              dircount = dircount + 1;
              cmd = ['mkdir ' output_dir current_filename_fh '/' sprintf('%.5d', dircount)];
              unix(cmd);
            end
            count=count+1;
              
            first_line = line;
            
            [t,r] = strtok(line);
            [id,r] = strtok(r);
            [secret,r] = strtok(r);
            [server,r] = strtok(r);
            
            line = fgetl(fid); %owner: line
            second_line = line;
            [t,r] = strtok(line);
            [owner,r] = strtok(r);
            
            %save all the metadata in the comment field of the file
            comment_field = strvcat(first_line, ... %photo id secret server
                                    second_line,... %owner
                                    fgetl(fid), ... %title
                                    fgetl(fid), ... %original_secret            
                                    fgetl(fid), ... %original_format
                                    fgetl(fid), ... %o_height
                                    fgetl(fid), ... %o_width
                                    fgetl(fid), ... %datetaken
                                    fgetl(fid), ... %dateupload
                                    fgetl(fid), ... %tags
                                    fgetl(fid), ... %license
                                    fgetl(fid), ... %latitude
                                    fgetl(fid), ... %longitude
                                    fgetl(fid), ... %accuracy
                                    fgetl(fid), ... %views
                                    fgetl(fid));    %interingness (string)
            
            %first lets try and grab the large size, which should be max
            %dimension = 1024 pixels (but there are bugs).
            url = ['http://static.flickr.com/' server '/' id '_' secret '_b.jpg'];
            
            fprintf('\n   current_image :  %s\n', [id '_' secret '_' server '_' owner '.jpg'] )
            
            %download the file to a temporary, local location
            %before saving it in the full path in order to minimize network
            %traffic using the /tmp/ space on any machine.
            
            %we want the file name to identify the image still.  not just be numbered.
            %use the -O [output file name] option
            %-t specifies number of retries
            %-T specifies all timeouts, in seconds.  if it times out does it retry?
            cmd = ['wget -t 3 -T 5 --quiet ' url ...
                   ' -O ' '/tmp/' id '_' secret '_' server '_' owner '.jpg' ];
               
            try
                unix(cmd);
            catch
                lasterr
                fprintf('XX!! Error with wget.\n');
            end

            %we need to check if we got a small error .gif back, in which case
            %we'll want to try for the original image.
            current_file_info = dir(['/tmp/' id '_' secret '_' server '_' owner '.jpg']);

            if(isempty(current_file_info))
                fprintf('XX!! could not find the temporary file from this iteration \n')
            else
                current_file_size = current_file_info.bytes;    
                
                if(current_file_size < 5000) %if the file is less than 20k, or we got an error .gif instead
                   fprintf('X  Large version did not exist, trying original\n');
                   %try for the original
                   url = ['http://static.flickr.com/' server '/' id '_' secret '_o.jpg'];
                   cmd = ['wget -t 3 -T 5 --quiet ' url ...
                             ' -O ' '/tmp/' id '_' secret '_' server '_' owner '.jpg' ];
                         
                   try
                       unix(cmd);
                   catch
                       lasterr
                       fprintf('XX!! Error with wget.\n');
                   end
                    
                   current_file_info = dir(['/tmp/' id '_' secret '_' server '_' owner '.jpg']);
                   if(isempty(current_file_info))
                       fprintf('XX!! could not find the second temporary file from this iteration \n')
                   else
                       current_file_size = current_file_info.bytes; 
                       
                       if(current_file_size < 5000) %if the file is less than 5k, or we got an error .gif instead
                           %neither the large nor the original existed
                           current_file_valid = 0;
                           fprintf('X  Original version does not exist\n');
                       else
                           %the large size did not exist, the original
                           %size did.  but it could be too small resolution.
                           %or too large, actually.
                           current_file_valid = 1;
                           fprintf('!  Original version exists\n');
                       end
                   end
                else
                    %the large size file existed and has enough bytes
                    %since it is large size, it's definitely high res
                    fprintf('!  Large version exists\n');
                    current_file_valid = 1;
                end

            
                if(current_file_valid == 1)
                    % load the image, resize it, remove border, save it.
                    try
                        current_image = imread( ['/tmp/' id '_' secret '_' server '_' owner '.jpg' ] );
                    catch
                        lasterr
                        fprintf('XX!! error loading temporary file, which should have been valid\n')
                    end

                    aspect_ratio = size(current_image,2) / size(current_image,1); %width by height
                    min_dim_pixels = min( size(current_image,1) , size(current_image,2) );

                    if(size(current_image,3) == 3 && ... %make sure it's color
                       aspect_ratio <= 1.6 && ...  %we want to allow 800x533 images, barely
                       aspect_ratio >= .625 && ... %we don't want massive images, they'll run matlab out of memory
                       min_dim_pixels >= 400 && ...
                       min_dim_pixels <= 1700)  %1700 min dimension largest allowable size.  This should almost NEVER happen, because if
                                                %the image had been this big then a 'large' size should have existed

                        current_image = double(current_image) /255;
                        current_image = remove_frame(current_image); %delete border

                        min_dim_pixels = min( size(current_image,1) , size(current_image,2) );
                        
                        if(min_dim_pixels >= 400)
                            %we finally have a completely valid image to save
                            
                            %resize the max dimension down to 1024
                            current_image = rescale_max_size(current_image, 1024, 1);
                            
                            output_filename = [output_dir current_filename_fh '/' sprintf('%.5d', dircount) '/' id '_' secret '_' server '_' owner '.jpg' ];

                            %lets save all the info we'll need in the comment
                            %section of the file.  We can retrieve this later
                            %with imfinfo()
                            try
                                imwrite(current_image, output_filename, 'jpg', 'quality', 85, 'comment', comment_field);
                                fprintf('!! Successfully wrote %s\n', output_filename)
                            catch
                                lasterr
                                fprintf('XX!! error writing final image\n')
                            end
                        else
                            %we deleted too many pixels from the border
                            fprintf('XX After border removal, the image is too small (%d pixels).\n', min_dim_pixels)
                        end
                    else
                        %print out the correct failure cases
                        if(min_dim_pixels < 400)
                            fprintf('XX Image is too small (%d pixels).  \n', min_dim_pixels)
                        end
                        
                        if(aspect_ratio < .625 || aspect_ratio > 1.6)
                            fprintf('XX Aspect ratio is bad (%f).\n', aspect_ratio)
                        end
                        
                        if(size(current_image,3) ~= 3)
                            fprintf('XX Image is not a 3 channel image (%d channels)\n', size(current_image,3))
                        end
                        
                        if(min_dim_pixels > 1700)
                            fprintf('XX The original image was huge, and flickr failed to make a large size\n');
                        end
                    end
                else
                    %this triggers less often than I would have thought.
                    fprintf('XX Could not find large or original size of this file. Skipping.\n');
                end
            end
            
            % delete the temporary file
            try
                delete(['/tmp/' id '_' secret '_' server '_' owner '.jpg' ]);
            catch
                lasterr
                fprintf('XX!! failed deleting the temporary file\n')
            end
            
            % pause inserted here so as not to piss flickr off...
            % maybe you can take it out?  probably, because the image
            % processing was slow enough.
            pause(1);
            
          end
        end
        
        fclose(fid);
        
    else
        fprintf(' file is locked / already downloaded, skipping\n')
    end
end %end of all files




