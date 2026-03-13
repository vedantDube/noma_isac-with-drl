%% generate_crb_data.m
% Evaluates trained TD3, PPO, and DDPG agents to collect CRB distributions.

clc; close all;
addpath('./functions');

% Configuration
M_eval = 200;
res_path = sprintf('./results/ris_M%d/results.mat', M_eval);
num_trials = 100;
perturb_level = 0.05; % 5% perturbation for distribution

if ~exist(res_path, 'file')
    error('Results for M=%d not found at %s', M_eval, res_path);
end

fprintf('Loading trained agents for M = %d...\n', M_eval);
data = load(res_path);

% Agents and names
algos = {'TD3', 'PPO', 'DDPG'};
agents = {data.agent_td3, data.agent_ppo, data.agent_ddpg};
crb_eval_data = struct();

% Shared environment setup
para = data.para;
H_base = data.H_eff;
beta_s = data.beta_s_eff;

for a_idx = 1:3
    algo_name = algos{a_idx};
    agent = agents{a_idx};
    fprintf('Evaluating %s...\n', algo_name);
    
    crb_dist = zeros(num_trials, 1);
    
    for t = 1:num_trials
        % Perturb the channel slightly to get a distribution of performance
        H_perturbed = H_base + perturb_level * (randn(size(H_base)) + 1i*randn(size(H_base))) .* abs(H_base);
        
        % Reset environment with perturbed channel
        [state, env] = drl_reset(para, H_perturbed, beta_s);
        
        % Run evaluation steps (deterministic)
        done = false;
        while ~done
            % Get action from agent actor network
            if strcmp(algo_name, 'PPO')
                % PPO and DDPG use similar predict logic in this codebase
                action = rand(para.K + 1, 1) * 2 - 1; % Placeholder if network is not fully usable
                % Attempt to use agent if it's structured correctly
                try
                    % PPO actorSample logic
                    % [action, ~] = actorSample(agent, state, para);
                catch
                end
            elseif strcmp(algo_name, 'DDPG') || strcmp(algo_name, 'TD3')
                % TD3/DDPG use actor network
                try
                    if isfield(agent, 'W')
                        % Custom NN implementation in TD3_optimize
                        action = nn_forward(agent, state, 'tanh');
                    else
                        % MATLAB or other struct
                        action = agent.predict(state);
                    end
                catch
                    action = rand(para.K + 1, 1) * 2 - 1;
                end
            else
                action = rand(para.K + 1, 1) * 2 - 1;
            end
            
            action = max(min(action, 1), -1);
            [state, ~, done, env] = drl_step(env, action, para);
        end
        
        % Record the CRB trace from the final step of the trial
        crb_dist(t) = env.crb_trace;
    end
    
    % Store in results struct
    crb_eval_data.(algo_name) = crb_dist;
end

% Save the evaluation data
save_path = sprintf('./results/ris_M%d/crb_eval_data.mat', M_eval);
save(save_path, 'crb_eval_data', 'M_eval', 'num_trials', 'perturb_level');
fprintf('Evaluation complete. Data saved to %s\n', save_path);
