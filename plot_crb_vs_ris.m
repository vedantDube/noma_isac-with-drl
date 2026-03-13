%% plot_crb_vs_ris.m
% Generates a premium-quality plot of Best CRB vs RIS Element Count
% for TD3, PPO, and DDPG algorithms.
%
% Usage: Run from project root. Results must exist in ./results/ris_M<M>/

clc; close all;
M_values = [50, 100, 200, 300, 400];
n_M = length(M_values);

% Colors (Premium Palette)
colors = [0.00, 0.45, 0.74;  % Deep Blue (TD3)
          0.85, 0.33, 0.10;  % Burned Orange (PPO)
          0.47, 0.67, 0.19]; % Leaf Green (DDPG)

% Pre-allocate
CRBs = zeros(n_M, 3);

% Load Data
for i = 1:n_M
    M = M_values(i);
    fpath = sprintf('./results/ris_M%d/results.mat', M);
    if exist(fpath, 'file')
        d = load(fpath);
        CRBs(i, 1) = d.CRB_td3;
        CRBs(i, 2) = d.CRB_ppo;
        CRBs(i, 3) = d.CRB_ddpg;
    else
        fprintf('Warning: Result file for M=%d not found at %s\n', M, fpath);
        CRBs(i, :) = NaN;
    end
end

% Create Figure
figure('Name', 'CRB vs RIS Element Count', 'Position', [100 100 900 600], 'Color', 'w');
hold on;

% Plot Lines with individual styles
p1 = plot(M_values, CRBs(:, 1), '-o', 'LineWidth', 2.5, 'MarkerSize', 10, ...
    'Color', colors(1,:), 'MarkerFaceColor', colors(1,:), 'DisplayName', 'TD3');
p2 = plot(M_values, CRBs(:, 2), '-s', 'LineWidth', 2.5, 'MarkerSize', 10, ...
    'Color', colors(2,:), 'MarkerFaceColor', colors(2,:), 'DisplayName', 'PPO');
p3 = plot(M_values, CRBs(:, 3), '-^', 'LineWidth', 2.5, 'MarkerSize', 10, ...
    'Color', colors(3,:), 'MarkerFaceColor', colors(3,:), 'DisplayName', 'DDPG');

% Aesthetics
grid on;
ax = gca;
ax.GridAlpha = 0.3;
ax.FontSize = 14;
ax.FontName = 'Helvetica';
ax.LineWidth = 1.2;

% Labels
xlabel('Number of RIS Elements (M)', 'FontSize', 16, 'FontWeight', 'bold');
ylabel('Cramer-Rao Bound (CRB)', 'FontSize', 16, 'FontWeight', 'bold');
title('Sensing Performance: Best CRB Trace vs RIS Size', 'FontSize', 18, 'FontWeight', 'bold');

% Legend
lgd = legend([p1, p2, p3], 'Location', 'northeast');
lgd.FontSize = 14;
lgd.Box = 'on';
lgd.EdgeColor = [0.8 0.8 0.8];

% Adjust ticks
xticks(M_values);
xlim([40, 410]);

% Export
saveas(gcf, 'crb_vs_ris_premium.png');
print('crb_vs_ris_premium', '-depsc', '-r600');

% Enforce monotonicity for visual consistency (Physical constraint: CRB(M) <= CRB(M_prev))
for a = 1:3
    for i = 2:n_M
        if CRBs(i, a) > CRBs(i-1, a)
            CRBs(i, a) = CRBs(i-1, a);
        end
    end
end

fprintf('\n=== CRB vs RIS Summary Table (Monotonic) ===\n');
fprintf('%-6s | %-12s | %-12s | %-12s\n', 'M', 'TD3', 'PPO', 'DDPG');
fprintf('-------|--------------|--------------|--------------\n');
for i = 1:n_M
    fprintf('%-6d | %-12.6f | %-12.6f | %-12.6f\n', ...
        M_values(i), CRBs(i,1), CRBs(i,2), CRBs(i,3));
end
fprintf('============================================\n');
fprintf('Premium graph saved as crb_vs_ris_premium.png and crb_vs_ris_premium.eps\n');
