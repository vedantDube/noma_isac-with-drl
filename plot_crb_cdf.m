%% plot_crb_cdf.m
% Generates a premium CDF vs CRB graph.

clc; close all;

% Configuration
M_plot = 200;
data_path = sprintf('./results/ris_M%d/crb_eval_data.mat', M_plot);

if ~exist(data_path, 'file')
    error('Evaluation data not found. Run generate_crb_data.m first.');
end

load(data_path);

% Algorithms and colors (matching project palette)
algos = {'TD3', 'PPO', 'DDPG'};
algo_colors = [0.2 0.5 0.9;   % TD3  – blue
               0.9 0.3 0.2;   % PPO  – red
               0.2 0.75 0.3]; % DDPG – green

h_fig = figure('Name', 'CRB CDF Comparison', 'Position', [100 100 800 600]);
hold on;

for i = 1:3
    algo_name = algos{i};
    crb_values = crb_eval_data.(algo_name);
    
    % Remove outliers/failed trials for a cleaner CDF if necessary
    crb_values(crb_values > 10) = 10; % Cap extremely high values for visibility
    
    [f, x] = ecdf(crb_values);
    
    % Premium line plotting
    plot(x, f, 'LineWidth', 3, 'Color', algo_colors(i,:));
end

hold off;

% Premium Styling
grid on;
box on;
set(gca, 'FontSize', 12, 'LineWidth', 1.2);
xlabel('CRB Trace (Sensing Performance)', 'FontSize', 14, 'FontWeight', 'bold');
ylabel('Cumulative Probability (CDF)', 'FontSize', 14, 'FontWeight', 'bold');
title(sprintf('CDF of CRB Performance (M = %d)', M_plot), 'FontSize', 16, 'FontWeight', 'bold');

legend(algos, 'Location', 'southeast', 'FontSize', 12, 'FontWeight', 'normal');

% Save the plot
output_name = 'crb_cdf_premium.png';
saveas(h_fig, output_name);
fprintf('CDF plot saved as %s\n', output_name);
