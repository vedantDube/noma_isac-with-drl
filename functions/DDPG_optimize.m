function [Rx_best, f_best, CRB_best, reward_history] = DDPG_optimize(para, H, beta_s, options, seed)
% DDPG (Deep Deterministic Policy Gradient) Optimizer for Near-Field ISAC
%  Deterministic policy gradient with actor and critic networks and replay buffer.
%  This implementation mirrors the TD3 code but uses a single critic and no policy noise.
%  Inputs: same as other optimizers; optional seed for reproducibility.

% optional RNG seed
if nargin >= 5 && ~isempty(seed)
    rng(seed,'twister');
end

% parse options
num_episodes = get_opt(options,'max_episodes',500);
max_steps    = get_opt(options,'max_steps',50);
hidden_dim   = get_opt(options,'hidden_dim',256);
lr_actor     = get_opt(options,'lr_actor',3e-4);
lr_critic    = get_opt(options,'lr_critic',3e-3);
gamma        = get_opt(options,'gamma',0.99);
tau          = get_opt(options,'tau',0.005);
batch_size   = get_opt(options,'batch_size',128);
buffer_size  = get_opt(options,'buffer_size',100000);
explore_noise= get_opt(options,'explore_noise',0.1);

K = para.K;
state_dim = K + 3;
action_dim= K + 1;

% actor and critic networks
actor = nn_init([state_dim, hidden_dim, hidden_dim, action_dim]);
critic= nn_init([state_dim + action_dim, hidden_dim, hidden_dim, 1]);
actor_tgt = actor;
critic_tgt= critic;
opt_a = adam_init(actor, lr_actor);
opt_c = adam_init(critic, lr_critic);

% replay buffer
buf.s = zeros(state_dim, buffer_size);
buf.a = zeros(action_dim, buffer_size);
buf.r = zeros(1, buffer_size);
buf.s2= zeros(state_dim, buffer_size);
buf.d = zeros(1, buffer_size);
buf.ptr=0; buf.cnt=0; buf.cap=buffer_size;

reward_history = zeros(num_episodes,1);
Rx_best = []; f_best = [];
CRB_best = inf;

fprintf('=== DDPG Training ===\n');
for ep = 1:num_episodes
    [state, env] = drl_reset(para, H, beta_s);
    total_reward = 0;
    done = false;
    steps=0;
    while ~done && steps < max_steps
        % select action with exploration noise
        [a_pred, cache_a] = nn_forward(actor, state, 'tanh');
        action = a_pred + explore_noise * randn(action_dim,1);
        action = max(min(action,1),-1);
        [next_state, reward, done, env] = drl_step(env, action, para);
        total_reward = total_reward + reward;
        buf = buf_store(buf, state, action, reward, next_state, done);
        state = next_state;
        steps = steps + 1;
        if buf.cnt >= batch_size
            [sb, ab, rb, s2b, db] = buf_sample(buf, batch_size);
            % critic update
            [a2, ~] = nn_forward(actor_tgt, s2b, 'tanh');
            q2 = nn_forward(critic_tgt, [s2b; a2], 'linear');
            y = rb + gamma * (1 - db) .* q2;
            [q_pred, cache_q] = nn_forward(critic, [sb; ab], 'linear');
            dq = 2 * (q_pred - y);
            [grads_c, ~] = nn_backward(critic, cache_q, dq, 'linear');
            [critic, opt_c] = adam_step(critic, grads_c, opt_c);
            % actor update (maximize Q)
            [a_pred2, cache_a2] = nn_forward(actor, sb, 'tanh');
            [q_pred2, cache_q2] = nn_forward(critic, [sb; a_pred2], 'linear');
            % gradient of -Q w.r.t. actor output
            dq_a = -ones(1,size(q_pred2,2)) / size(q_pred2,2);
            [~, dx_q2] = nn_backward(critic, cache_q2, dq_a, 'linear');
            da = dx_q2(state_dim+1:end, :);
            [grads_a, ~] = nn_backward(actor, cache_a2, da, 'tanh');
            [actor, opt_a] = adam_step(actor, grads_a, opt_a);
            % soft updates
            actor_tgt  = soft_update(actor_tgt,  actor,  tau);
            critic_tgt = soft_update(critic_tgt, critic, tau);
        end
    end
    reward_history(ep) = total_reward;
    if isfield(env, 'best_CRB') && env.best_CRB < CRB_best
        CRB_best = env.best_CRB;
        Rx_best = env.best_Rx;
        f_best  = env.best_f;
    end
    if mod(ep, 20) == 0
        fprintf('Episode %3d/%d | Reward: %.2f | Best CRB: %.6f\n', ep, num_episodes, total_reward, CRB_best);
    end
end

% final evaluation on clean channel
fprintf('\nFinal evaluation on clean channel...\n');
[state, env] = drl_reset(para, H, beta_s);
done=false;
while ~done
    [a_pred, ~] = nn_forward(actor, state, 'tanh');
    [state, ~, done, env] = drl_step(env, a_pred, para);
end
if env.best_CRB < CRB_best
    CRB_best = env.best_CRB;
    Rx_best  = env.best_Rx;
    f_best   = env.best_f;
end
fprintf('DDPG Final CRB Trace: %.6f\n', CRB_best);

%%% helper functions (copied from TD3) %%%

function net = nn_init(dims)
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
    cache.h = cell(1, net.L + 1);
    cache.z = cell(1, net.L);
    cache.h{1} = x;
    for l = 1:net.L
        cache.z{l} = bsxfun(@plus, net.W{l} * cache.h{l}, net.b{l});
        if l < net.L
            cache.h{l+1} = max(0, cache.z{l});
        else
            if strcmp(out_act, 'tanh')
                cache.h{l+1} = tanh(cache.z{l});
            else
                cache.h{l+1} = cache.z{l};
            end
        end
    end
    out = cache.h{end};
end

function [grads, dx] = nn_backward(net, cache, dout, out_act)
    B = size(dout, 2);
    grads.dW = cell(1, net.L);
    grads.db = cell(1, net.L);
    dl = dout;
    for l = net.L:-1:1
        if l == net.L && strcmp(out_act, 'tanh')
            dl = dl .* (1 - cache.h{l+1}.^2);
        elseif l < net.L
            dl = dl .* (cache.z{l} > 0);
        end
        grads.dW{l} = (dl * cache.h{l}') / B;
        grads.db{l} = sum(dl, 2) / B;
        dl = net.W{l}' * dl;
    end
    dx = dl;
end

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
        opt.mW{l} = opt.beta1*opt.mW{l} + (1-opt.beta1)*grads.dW{l};
        opt.vW{l} = opt.beta2*opt.vW{l} + (1-opt.beta2)*(grads.dW{l}.^2);
        mW_hat = opt.mW{l} / (1 - opt.beta1^opt.t);
        vW_hat = opt.vW{l} / (1 - opt.beta2^opt.t);
        net.W{l} = net.W{l} - opt.lr * mW_hat ./ (sqrt(vW_hat)+1e-8);
        opt.mb{l} = opt.beta1*opt.mb{l} + (1-opt.beta1)*grads.db{l};
        opt.vb{l} = opt.beta2*opt.vb{l} + (1-opt.beta2)*(grads.db{l}.^2);
        mb_hat = opt.mb{l} / (1 - opt.beta1^opt.t);
        vb_hat = opt.vb{l} / (1 - opt.beta2^opt.t);
        net.b{l} = net.b{l} - opt.lr * mb_hat ./ (sqrt(vb_hat)+1e-8);
    end
end

function buf = buf_store(buf, s, a, r, s2, d)
    buf.ptr = mod(buf.ptr, buf.cap) + 1;
    buf.cnt = min(buf.cnt + 1, buf.cap);
    idx = buf.ptr;
    buf.s(:,idx) = s;
    buf.a(:,idx) = a;
    buf.r(idx) = r;
    buf.s2(:,idx)= s2;
    buf.d(idx) = d;
end

function [sb, ab, rb, s2b, db] = buf_sample(buf, batch_size)
    idx = randperm(buf.cnt, batch_size);
    sb = buf.s(:,idx);
    ab = buf.a(:,idx);
    rb = buf.r(idx);
    s2b= buf.s2(:,idx);
    db = buf.d(idx);
end

function net_tgt = soft_update(net_tgt, net, tau)
    for l = 1:net.L
        net_tgt.W{l} = (1-tau)*net_tgt.W{l} + tau*net.W{l};
        net_tgt.b{l} = (1-tau)*net_tgt.b{l} + tau*net.b{l};
    end
end
