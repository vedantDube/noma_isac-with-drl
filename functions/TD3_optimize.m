function [Rx_best, f_best, CRB_best, reward_history] = TD3_optimize(para, H, beta_s, options, seed)
%TD3 (Twin Delayed DDPG) Optimizer for Near-Field ISAC
%   [Rx_best, f_best, CRB_best, reward_history] = TD3_optimize(para, H, beta_s, options[, seed])
%   optional seed argument makes the training deterministic by seeding
%   MATLAB's RNG at the beginning of the call.
%
% Uses Deep Reinforcement Learning (TD3) to minimize the CRB for robust
% sensing while satisfying communication rate constraints. The DRL agent
% learns a policy that produces beamforming solutions robust to channel
% noise and model uncertainties.
%
% TD3 is an actor-critic algorithm with:
%   - Deterministic actor network (state -> action)
%   - Twin critic networks (state, action -> Q-value) to reduce overestimation
%   - Delayed policy updates for stability
%   - Target networks with soft updates
%
% Inputs:
%   para    - system parameters (from para_init)
%   H       - communication channel matrix [N x K]
%   beta_s  - complex sensing channel gain
%   options - (optional) struct with training hyperparameters:
%             .num_episodes  (default: 200)
%             .hidden_dim    (default: 128)
%             .lr_actor      (default: 1e-4)
%             .lr_critic     (default: 1e-3)
%             .gamma         (default: 0.99)  discount factor
%             .tau           (default: 0.005) soft update rate
%             .policy_noise  (default: 0.2)   target policy noise
%             .noise_clip    (default: 0.5)   noise clipping
%             .policy_delay  (default: 2)     actor update frequency
%             .batch_size    (default: 64)
%             .buffer_size   (default: 50000)
%             .warmup_steps  (default: 500)
%             .explore_noise (default: 0.1)   exploration noise std
%             .channel_noise (default: 0.01)  channel perturbation for robustness
%
% Outputs:
%   Rx_best        - best transmit covariance matrix [N x N]
%   f_best         - best beamforming vectors [N x K]
%   CRB_best       - best CRB trace achieved (scalar)
%   reward_history - total reward per episode [num_episodes x 1]

    % optionally seed the random number generator for reproducibility
    if nargin >= 5 && ~isempty(seed)
        rng(seed, 'twister');
    end

    K = para.K;
    state_dim = K + 3;             % rates(K) + CRB + sensing_frac + step_progress
    action_dim = K + 1;            % power weights(K) + sensing fraction

    %% Parse options with defaults
    if nargin < 4, options = struct(); end
    num_episodes  = get_opt(options, 'num_episodes',  500);
    hidden_dim    = get_opt(options, 'hidden_dim',    256);
    lr_actor      = get_opt(options, 'lr_actor',      3e-4);
    lr_critic     = get_opt(options, 'lr_critic',     3e-4);
    gamma         = get_opt(options, 'gamma',         0.99);
    tau           = get_opt(options, 'tau',            0.005);
    pol_noise     = get_opt(options, 'policy_noise',   0.2);
    noise_clip    = get_opt(options, 'noise_clip',     0.5);
    pol_delay     = get_opt(options, 'policy_delay',   2);
    batch_size    = get_opt(options, 'batch_size',     128);
    buffer_size   = get_opt(options, 'buffer_size',    100000);
    warmup_steps  = get_opt(options, 'warmup_steps',   1000);
    explore_noise = get_opt(options, 'explore_noise',  0.15);
    ch_noise      = get_opt(options, 'channel_noise',  0.02);

    %% Initialize Actor and Twin Critic Networks
    % Actor: state -> action (tanh output, bounded [-1,1])
    actor  = nn_init([state_dim, hidden_dim, hidden_dim, action_dim]);
    % Critic 1 & 2: (state, action) -> Q-value (linear output)
    critic1 = nn_init([state_dim + action_dim, hidden_dim, hidden_dim, 1]);
    critic2 = nn_init([state_dim + action_dim, hidden_dim, hidden_dim, 1]);

    % Target networks (identical copies, updated slowly)
    actor_tgt   = actor;
    critic1_tgt = critic1;
    critic2_tgt = critic2;

    % Adam optimizers for each network
    opt_a  = adam_init(actor,   lr_actor);
    opt_c1 = adam_init(critic1, lr_critic);
    opt_c2 = adam_init(critic2, lr_critic);

    %% Replay Buffer
    buf.s   = zeros(state_dim,  buffer_size);
    buf.a   = zeros(action_dim, buffer_size);
    buf.r   = zeros(1, buffer_size);
    buf.s2  = zeros(state_dim,  buffer_size);
    buf.d   = zeros(1, buffer_size);
    buf.ptr = 0;
    buf.cnt = 0;
    buf.cap = buffer_size;

    %% Training Loop
    reward_history = zeros(num_episodes, 1);
    total_steps = 0;
    Rx_best  = [];
    f_best   = [];
    CRB_best = inf;

    fprintf('=== TD3 Training ===\n');
    fprintf('Episodes: %d | State dim: %d | Action dim: %d\n', ...
            num_episodes, state_dim, action_dim);
    fprintf('Hidden: %d | Batch: %d | Buffer: %d\n\n', ...
            hidden_dim, batch_size, buffer_size);

    for ep = 1:num_episodes
        % Anneal exploration noise: linear decay from explore_noise to 0.02
        eps_frac = (ep - 1) / max(num_episodes - 1, 1);
        cur_noise = explore_noise * (1 - 0.85 * eps_frac);  % decays to 15% of initial

        % Add noise to channel for robustness training (decay channel noise too)
        cur_ch_noise = ch_noise * (1 - 0.5 * eps_frac);
        H_noisy = H + cur_ch_noise * (randn(size(H)) + 1i*randn(size(H))) .* abs(H);
        [state, env] = drl_reset(para, H_noisy, beta_s);

        episode_reward = 0;
        done = false;

        while ~done
            total_steps = total_steps + 1;

            % Select action
            if total_steps <= warmup_steps
                % Random exploration during warmup
                action = 2 * rand(action_dim, 1) - 1;
            else
                % Actor policy + annealed exploration noise
                action = nn_forward(actor, state, 'tanh');
                action = action + cur_noise * randn(action_dim, 1);
                action = max(min(action, 1), -1);
            end

            % Execute action in environment
            [next_state, reward, done, env] = drl_step(env, action, para);
            episode_reward = episode_reward + reward;

            % Store transition in replay buffer
            buf = buf_store(buf, state, action, reward, next_state, done);
            state = next_state;

            % Train networks if enough samples collected
            if buf.cnt >= batch_size && total_steps > warmup_steps
                % Sample mini-batch
                [sb, ab, rb, s2b, db] = buf_sample(buf, batch_size);

                %% ---- Critic Update ----
                % Compute target actions with clipped noise (smoothing)
                next_a = nn_forward(actor_tgt, s2b, 'tanh');
                tn = randn(size(next_a)) * pol_noise;
                tn = max(min(tn, noise_clip), -noise_clip);
                next_a = max(min(next_a + tn, 1), -1);

                % Target Q-values: y = r + gamma * (1-d) * min(Q1_tgt, Q2_tgt)
                q1t = nn_forward(critic1_tgt, [s2b; next_a], 'linear');
                q2t = nn_forward(critic2_tgt, [s2b; next_a], 'linear');
                y = rb + gamma * (1 - db) .* min(q1t, q2t);

                % Update Critic 1: minimize MSE(Q1 - y)
                sa = [sb; ab];
                [q1, cache1] = nn_forward(critic1, sa, 'linear');
                dq1 = 2 * (q1 - y);
                [grads1, ~] = nn_backward(critic1, cache1, dq1, 'linear');
                [critic1, opt_c1] = adam_step(critic1, grads1, opt_c1);

                % Update Critic 2: minimize MSE(Q2 - y)
                [q2, cache2] = nn_forward(critic2, sa, 'linear');
                dq2 = 2 * (q2 - y);
                [grads2, ~] = nn_backward(critic2, cache2, dq2, 'linear');
                [critic2, opt_c2] = adam_step(critic2, grads2, opt_c2);

                %% ---- Delayed Actor Update ----
                if mod(total_steps, pol_delay) == 0
                    % Actor loss: maximize Q1(s, actor(s)) -> minimize -Q1
                    [a_pred, cache_a] = nn_forward(actor, sb, 'tanh');
                    [~, cache_c] = nn_forward(critic1, [sb; a_pred], 'linear');

                    % Backprop through critic to get dQ/da
                    dq = -ones(1, batch_size) / batch_size;
                    [~, dx_c] = nn_backward(critic1, cache_c, dq, 'linear');

                    % Extract gradient w.r.t. action (last action_dim rows)
                    da = dx_c(state_dim+1:end, :);

                    % Backprop through actor
                    [grads_a, ~] = nn_backward(actor, cache_a, da, 'tanh');
                    [actor, opt_a] = adam_step(actor, grads_a, opt_a);

                    % Soft update target networks
                    actor_tgt   = soft_update(actor,   actor_tgt,   tau);
                    critic1_tgt = soft_update(critic1, critic1_tgt, tau);
                    critic2_tgt = soft_update(critic2, critic2_tgt, tau);
                end
            end
        end

        reward_history(ep) = episode_reward;

        % Track global best solution
        if isfield(env, 'best_CRB') && env.best_CRB < CRB_best
            CRB_best = env.best_CRB;
            Rx_best  = env.best_Rx;
            f_best   = env.best_f;
        end

        if mod(ep, 20) == 0
            fprintf('Episode %3d/%d | Reward: %8.2f | Best CRB: %.6f\n', ...
                    ep, num_episodes, episode_reward, CRB_best);
        end
    end

    %% Final evaluation on clean (noise-free) channel
    fprintf('\nFinal evaluation on clean channel...\n');
    [state, env] = drl_reset(para, H, beta_s);
    done = false;
    while ~done
        action = nn_forward(actor, state, 'tanh');
        [state, ~, done, env] = drl_step(env, action, para);
    end
    if env.best_CRB < CRB_best
        CRB_best = env.best_CRB;
        Rx_best  = env.best_Rx;
        f_best   = env.best_f;
    end
    fprintf('DRL Final CRB Trace: %.6f\n', CRB_best);
end


%% ==================== LOCAL FUNCTIONS ====================

%% ---------- Neural Network ----------

function net = nn_init(dims)
% Initialize fully-connected network with Xavier initialization
% dims = [input_dim, hidden1, hidden2, ..., output_dim]
    L = length(dims) - 1;
    net.W = cell(1, L);
    net.b = cell(1, L);
    net.L = L;
    for l = 1:L
        net.W{l} = randn(dims(l+1), dims(l)) * sqrt(2 / dims(l));
        net.b{l} = zeros(dims(l+1), 1);
    end
end

function [out, cache] = nn_forward(net, x, out_act)
% Forward pass: ReLU hidden layers, out_act ('tanh'/'linear') for output
% x can be [dim x 1] (single) or [dim x batch] (batch)
    cache.h = cell(1, net.L + 1);
    cache.z = cell(1, net.L);
    cache.h{1} = x;

    for l = 1:net.L
        cache.z{l} = bsxfun(@plus, net.W{l} * cache.h{l}, net.b{l});
        if l < net.L
            cache.h{l+1} = max(0, cache.z{l});      % ReLU
        else
            if strcmp(out_act, 'tanh')
                cache.h{l+1} = tanh(cache.z{l});     % bounded [-1,1]
            else
                cache.h{l+1} = cache.z{l};            % linear
            end
        end
    end
    out = cache.h{end};
end

function [grads, dx] = nn_backward(net, cache, dout, out_act)
% Backward pass: compute gradients for weights/biases and input gradient dx
    B = size(dout, 2);
    grads.dW = cell(1, net.L);
    grads.db = cell(1, net.L);

    dl = dout;
    for l = net.L:-1:1
        % Apply activation derivative
        if l == net.L
            if strcmp(out_act, 'tanh')
                dl = dl .* (1 - cache.h{l+1}.^2);   % tanh derivative
            end
        else
            dl = dl .* (cache.z{l} > 0);              % ReLU derivative
        end

        grads.dW{l} = (dl * cache.h{l}') / B;
        grads.db{l} = sum(dl, 2) / B;
        dl = net.W{l}' * dl;
    end
    dx = dl;
end


%% ---------- Adam Optimizer ----------

function opt = adam_init(net, lr)
    opt.lr    = lr;
    opt.beta1 = 0.9;
    opt.beta2 = 0.999;
    opt.t     = 0;
    opt.mW = cell(1, net.L);
    opt.vW = cell(1, net.L);
    opt.mb = cell(1, net.L);
    opt.vb = cell(1, net.L);
    for l = 1:net.L
        opt.mW{l} = zeros(size(net.W{l}));
        opt.vW{l} = zeros(size(net.W{l}));
        opt.mb{l} = zeros(size(net.b{l}));
        opt.vb{l} = zeros(size(net.b{l}));
    end
end

function [net, opt] = adam_step(net, grads, opt)
    opt.t = opt.t + 1;
    for l = 1:net.L
        % Update weights
        opt.mW{l} = opt.beta1 * opt.mW{l} + (1 - opt.beta1) * grads.dW{l};
        opt.vW{l} = opt.beta2 * opt.vW{l} + (1 - opt.beta2) * grads.dW{l}.^2;
        mh = opt.mW{l} / (1 - opt.beta1^opt.t);
        vh = opt.vW{l} / (1 - opt.beta2^opt.t);
        net.W{l} = net.W{l} - opt.lr * mh ./ (sqrt(vh) + 1e-8);

        % Update biases
        opt.mb{l} = opt.beta1 * opt.mb{l} + (1 - opt.beta1) * grads.db{l};
        opt.vb{l} = opt.beta2 * opt.vb{l} + (1 - opt.beta2) * grads.db{l}.^2;
        mh = opt.mb{l} / (1 - opt.beta1^opt.t);
        vh = opt.vb{l} / (1 - opt.beta2^opt.t);
        net.b{l} = net.b{l} - opt.lr * mh ./ (sqrt(vh) + 1e-8);
    end
end


%% ---------- Replay Buffer ----------

function buf = buf_store(buf, s, a, r, s2, d)
    idx = mod(buf.ptr, buf.cap) + 1;
    buf.s(:, idx)  = s;
    buf.a(:, idx)  = a;
    buf.r(idx)     = r;
    buf.s2(:, idx) = s2;
    buf.d(idx)     = d;
    buf.ptr = buf.ptr + 1;
    buf.cnt = min(buf.cnt + 1, buf.cap);
end

function [s, a, r, s2, d] = buf_sample(buf, n)
    idx = randperm(buf.cnt, n);
    s  = buf.s(:, idx);
    a  = buf.a(:, idx);
    r  = buf.r(idx);
    s2 = buf.s2(:, idx);
    d  = buf.d(idx);
end


%% ---------- Soft Target Update ----------

function tgt = soft_update(net, tgt, tau)
    for l = 1:net.L
        tgt.W{l} = tau * net.W{l} + (1 - tau) * tgt.W{l};
        tgt.b{l} = tau * net.b{l} + (1 - tau) * tgt.b{l};
    end
end


%% ---------- Utility ----------

function val = get_opt(s, name, default)
    if isfield(s, name)
        val = s.(name);
    else
        val = default;
    end
end
