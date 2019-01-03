% ****** HW5 EM Algorithm ******* %

function [mean_result, variace_result, prior_result] = EM(dataset, c, mean, covariance, prior)
    [size_x size_y] = size(dataset);
    clear size_y
    
    % hij matrix from the slides
    hij = zeros(size_x, c);

    %z = diag(ones(c,1));
    
    prev_prior = prior; cur_prior = prior;
    prev_mean = mean; cur_mean = mean;
    var_prev = covariance; var_cur = covariance;

    for iterations = 1 : 100

        % E - STEP for Gaussian Mixtures, from the slides
        for row = 1 : size_x
            for component = 1:c
                hij(row, component) =  prev_prior(component) * mvnpdf(dataset(row, :), prev_mean(component, :), diag(var_prev(component, :)));
            end
            sum_hij = sum(hij(row, :));
            hij(row, :) = hij(row, :) / sum_hij;
        end

        % M -  STEP for Gaussian Mixtures, from the slides, Lagrangian
        % formulation was used in the slides to satisfy the constraint sum(pij)equal to 1
        for j = 1:c
            mj_raw = zeros(1,64);
            
            %Alternative solution, takes more time
			% mj_raw = zeros(1,64);
			% for i = 1:size_x
			% mj_raw = mj_raw + hij(i,j) * dataset(i,:);
			% end
			
			%Calculating the next priors and mean
			row_sum = sum(hij(:, j));
			mj_raw = hij(:, j)'*dataset(:, :);

            %Calculating the next priors and mean
            row_sum = sum(hij(:, j));

            %New means
            cur_mean(j, :) = mj_raw./row_sum;
            %New priors
            cur_prior(j) = row_sum / size_x;

        end

        % This part is seperated from above for easier debugging
        % Should be merged with the above part
        
        for j = 1:c
            
           sigmaj_raw = 0;

           for i=1: size(dataset,1)
               square_diff = (dataset(i,:)-cur_mean(j,:)).^2;
               sigmaj_raw = sigmaj_raw + hij(i,j) * square_diff;
           end
           row_sum = sum(hij(:, j));
           %New covariance below
           var_cur(j, :) = sigmaj_raw / row_sum;
           %Regularizing the covariance to prevent possible problems from 0 values
           var_cur(var_cur < 0.0005) = 0.0005;
        end
        %For test purposes
%         value = abs(cur_mean - prev_mean);
%         condition = mean(mean(value));
        prev_mean = cur_mean;
        prev_prior = cur_prior;
        var_prev = var_cur;  

    end

    prior_result = prev_prior;
    mean_result = prev_mean;
    variace_result = var_prev;