%% plot_crb_cdf_export.m
% Generates separate EPS plots for M=100 and M=400 and updates labels.

clc; close all;

M_export = [100, 400];
algos = {'TD3', 'PPO', 'DDPG'};
algo_colors = [0.2 0.5 0.9;   % TD3  – blue
               0.9 0.3 0.2;   % PPO  – red
               0.2 0.75 0.3]; % DDPG – green

for M = M_export
    data_path = sprintf('./results/ris_M%d/crb_eval_data.mat', M);
    
    if exist(data_path, 'file')
        load(data_path);
        
        h_fig = figure('Name', sprintf('CRB CDF M=%d', M), 'Position', [100 100 600 500]);
        hold on;
        
        for a = 1:3
            algo_name = algos{a};
            crb_values = crb_eval_data.(algo_name);
            crb_values(crb_values > 10) = 10; % Cap
            
            [f, x] = ecdf(crb_values);
            plot(x, f, 'LineWidth', 2.5, 'Color', algo_colors(a,:));
        end
        
        hold off;
        grid on; box on;
        set(gca, 'FontSize', 12);
        
        % Updated labels as per user request
        xlabel('CRB Trace (Sensing Performance)', 'FontSize', 14, 'FontWeight', 'bold');
        ylabel('Cumulative Probability (CDF)', 'FontSize', 14, 'FontWeight', 'bold');
        title(sprintf('CDF of Performance (RIS M = %d)', M), 'FontSize', 16, 'FontWeight', 'bold');
        legend(algos, 'Location', 'southeast', 'FontSize', 11);
        
        % Export as EPS
        eps_filename = sprintf('crb_cdf_M%d.eps', M);
        exportgraphics(h_fig, eps_filename, 'ContentType', 'vector');
        saveas(h_fig, sprintf('crb_cdf_M%d.png', M)); % Also save png for preview
        fprintf('Saved %s and crb_cdf_M%d.png\n', eps_filename, M);
    else
        fprintf('Data for M=%d not found. Run generate_crb_data_all.m first.\n', M);
    end
end
