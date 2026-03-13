% Compare TD3, PPO, DDPG reward vs episode
para = para_init();
H = generate_channel();
beta_s = 1; % Example value
options.max_episodes = 100;

[~, ~, ~, td3_reward] = TD3_optimize(para, H, beta_s, options);
[~, ~, ~, ppo_reward] = PPO_optimize(para, H, beta_s, options);
[~, ~, ~, ddpg_reward] = DDPG_optimize(para, H, beta_s, options);

figure;
hold on;
plot(td3_reward, 'LineWidth', 2);
plot(ppo_reward, 'LineWidth', 2);
plot(ddpg_reward, 'LineWidth', 2);
hold off;
legend('TD3', 'PPO', 'DDPG');
xlabel('Episode');
ylabel('Reward');
title('Reward vs Episode Comparison');
grid on;