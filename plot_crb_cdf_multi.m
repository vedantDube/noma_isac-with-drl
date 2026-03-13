%% plot_crb_cdf_multi.m
% Generates combined CDF vs CRB graphs for all RIS sizes (M).

clc; close all;

M_list = [50, 100, 200, 300, 400];
algos = {'TD3', 'PPO', 'DDPG'};
algo_colors = [0.2 0.5 0.9;   % TD3  – blue
               0.9 0.3 0.2;   % PPO  – red
               0.2 0.75 0.3]; % DDPG – green

h_fig = figure('Name', 'Multi-M CRB CDF Comparison', 'Position', [50 50 1400 800]);

for i = 1:length(M_list)
    M = M_list(i);
    data_path = sprintf('./results/ris_M%d/crb_eval_data.mat', M);
    
    subplot(2, 3, i);
    hold on;
    
    if exist(data_path, 'file')
        load(data_path);
        for a = 1:3
            algo_name = algos{a};
            crb_values = crb_eval_data.(algo_name);
            crb_values(crb_values > 10) = 10; % Cap for visibility
            
            [f, x] = ecdf(crb_values);
            plot(x, f, 'LineWidth', 2.5, 'Color', algo_colors(a,:));
        end
    else
        text(0.5, 0.5, sprintf('Data for M=%d missing', M), 'HorizontalAlignment', 'center');
    end
    
    hold off;
    grid on; box on;
    xlabel('CRB Trace (Sumrate)', 'FontSize', 10);
    ylabel('CDF', 'FontSize', 10);
    title(sprintf('RIS M = %d', M), 'FontSize', 12, 'FontWeight', 'bold');
    if i == 1
        legend(algos, 'Location', 'southeast', 'FontSize', 8);
    end
end

sgtitle('CRB Performance CDF for Different RIS Sizes', 'FontSize', 16, 'FontWeight', 'bold');

saveas(h_fig, 'crb_cdf_multi_premium.png');
fprintf('Multi-M CDF plot saved as crb_cdf_multi_premium.png\n');
