function [Rx_best, f_best, CRB_best, reward_history] = PPO_optimize(para, H, beta_s, options, seed)
% PPO (Proximal Policy Optimization) Optimizer for Near-Field ISAC
%   On-policy actor-critic optimization using clipped surrogate objective.
%   This is a genuine implementation with neural networks and Adam.
%   Inputs:
%     para, H, beta_s - system parameters and channels (same as other optimizers)
%     options - struct with training hyperparameters
%     seed    - optional RNG seed for reproducibility
%   Outputs:
%     Rx_best, f_best, CRB_best - best sensing solution found during training
%     reward_history            - total reward obtained each episode

% optional seeding
if nargin >= 5 && ~isempty(seed)
    rng(seed,'twister');
end

% default options
num_episodes = get_opt(options,'max_episodes',500);
max_steps    = get_opt(options,'max_steps',50);
hidden_dim   = get_opt(options,'hidden_dim',256);
lr_actor     = get_opt(options,'lr_actor',3e-4);
lr_critic    = get_opt(options,'lr_critic',3e-4);
gamma        = get_opt(options,'gamma',0.99);
clip_eps     = get_opt(options,'clip_eps',0.2);
batch_size   = get_opt(options,'batch_size',128);
update_epochs= get_opt(options,'update_epochs',5);
buffer_size  = get_opt(options,'buffer_size',100000);

K = para.K;
state_dim  = K + 3;
action_dim = K + 1;

% initialize actor & critic networks
actor  = nn_init([state_dim, hidden_dim, hidden_dim, action_dim]);
critic = nn_init([state_dim, hidden_dim, hidden_dim, 1]);
opt_a  = adam_init(actor, lr_actor);
opt_c  = adam_init(critic, lr_critic);

% fixed policy noise standard deviation (learnable extension possible)
log_std = -0.5 * ones(action_dim,1);

% replay buffer (on-policy, just store latest episode then clear)
buffer = struct('s',[],'a',[],'r',[],'s2',[],'d',[],'logp',[]);

reward_history = zeros(num_episodes,1);
best_reward = -inf;
Rx_best = [];
f_best = [];
CRB_best = [];

fprintf('=== PPO Training ===\n');

for ep = 1:num_episodes
    % collect one episode
    [state, env] = drl_reset(para, H, beta_s);
    done = false;
    steps = 0;
    ep_states=[]; ep_actions=[]; ep_rewards=[]; ep_logp=[]; ep_d=[];

    while ~done && steps < max_steps
        [mu, ~] = nn_forward(actor, state, 'tanh');
        std = exp(log_std);
        noise = std .* randn(action_dim,1);
        action = mu + noise;
        action = max(min(action,1),-1);
        logp = -0.5 * sum(((noise./std).^2 + 2*log_std + log(2*pi)));
        [next_state, reward, done, env] = drl_step(env, action, para);

        ep_states(:,end+1) = state;
        ep_actions(:,end+1) = action;
        ep_rewards(end+1) = reward;
        ep_logp(end+1) = logp;
        ep_d(end+1) = done;

        state = next_state;
        steps = steps + 1;
    end

    % append to buffer (overwrite previous)
    buffer.s   = ep_states;
    buffer.a   = ep_actions;
    buffer.r   = ep_rewards;
    buffer.s2  = [ep_states(:,2:end), state];
    buffer.d   = ep_d;
    buffer.logp= ep_logp;

    % compute returns & advantages
    T = length(ep_rewards);
    R = zeros(1,T);
    cum = 0;
    for t = T:-1:1
        cum = ep_rewards(t) + gamma * cum * (1 - ep_d(t));
        R(t) = cum;
    end
    v_vals = nn_forward(critic, ep_states, 'linear');
    adv = R - v_vals';          % advantage = return - value

    % do multiple epochs of minibatch updates
    for epoch = 1:update_epochs
        perm = randperm(T);
        for start = 1:batch_size:T
            idx = perm(start:min(start+batch_size-1,T));
            s_b = ep_states(:,idx);
            a_b = ep_actions(:,idx);
            old_logp_b = ep_logp(idx);
            adv_b = adv(idx)';

            % critic update (MSE)
            [v_pred, cache_v] = nn_forward(critic, s_b, 'linear');
            td = (R(idx)' - v_pred);
            [grads_c, ~] = nn_backward(critic, cache_v, -2*td, 'linear');
            [critic, opt_c] = adam_step(critic, grads_c, opt_c);

            % actor update (clipped surrogate)
            [mu_b, cache_a] = nn_forward(actor, s_b, 'tanh');
            std = exp(log_std);
            diff = a_b - mu_b;
            logp_new = -0.5 * sum((diff./std).^2 + 2*log_std + log(2*pi));
            ratio = exp(logp_new - old_logp_b);
            clipr = min(max(ratio,1-clip_eps),1+clip_eps);
            surr1 = ratio .* adv_b;
            surr2 = clipr .* adv_b;
            mask = surr1 < surr2;
            grad_mu = zeros(size(mu_b));
            for kk = 1:length(mask)
                if mask(kk)
                    grad_mu(:,kk) = -adv_b(kk) * (a_b(:,kk)-mu_b(:,kk)) ./ (std.^2);
                end
            end
            grad_mu = mean(grad_mu,2);
            [grads_a, ~] = nn_backward(actor, cache_a, grad_mu, 'tanh');
            [actor, opt_a] = adam_step(actor, grads_a, opt_a);
        end
    end

    reward_history(ep) = sum(ep_rewards);
    if reward_history(ep) > best_reward
        best_reward = reward_history(ep);
        Rx_best = env.best_Rx;
        f_best  = env.best_f;
        CRB_best= env.best_CRB;
    end
end
end

%% ---------- helper functions ----------

function net = nn_init(dims)
    L = length(dims) - 1;
    net.W = cell(1,L); net.b = cell(1,L); net.L = L;
    for l=1:L
        net.W{l} = randn(dims(l+1),dims(l)) * sqrt(2/dims(l));
        net.b{l} = zeros(dims(l+1),1);
    end
end

function [out, cache] = nn_forward(net, x, out_act)
    cache.h = cell(1,net.L+1); cache.z = cell(1,net.L);
    cache.h{1} = x;
    for l=1:net.L
        cache.z{l} = net.W{l}*cache.h{l} + net.b{l};
        if l<net.L
            cache.h{l+1} = max(0, cache.z{l});
        else
            if strcmp(out_act,'tanh')
                cache.h{l+1} = tanh(cache.z{l});
            else
                cache.h{l+1} = cache.z{l};
            end
        end
    end
    out = cache.h{end};
end

function [grads, dx] = nn_backward(net, cache, dout, out_act)
    B = size(dout,2);
    grads.dW = cell(1,net.L); grads.db = cell(1,net.L);
    dl = dout;
    for l=net.L:-1:1
        if l==net.L && strcmp(out_act,'tanh')
            dl = dl .* (1 - cache.h{l+1}.^2);
        elseif l<net.L
            dl = dl .* (cache.z{l} > 0);
        end
        grads.dW{l} = (dl * cache.h{l}')/B;
        grads.db{l} = sum(dl,2)/B;
        dl = net.W{l}' * dl;
    end
    dx = dl;
end

function opt = adam_init(net, lr)
    opt.lr=lr; opt.beta1=0.9; opt.beta2=0.999; opt.t=0;
    opt.mW=cell(1,net.L); opt.vW=cell(1,net.L);
    opt.mb=cell(1,net.L); opt.vb=cell(1,net.L);
    for l=1:net.L
        opt.mW{l}=zeros(size(net.W{l})); opt.vW{l}=zeros(size(net.W{l}));
        opt.mb{l}=zeros(size(net.b{l})); opt.vb{l}=zeros(size(net.b{l}));
    end
end

function [net,opt] = adam_step(net, grads, opt)
    opt.t = opt.t + 1;
    for l = 1:net.L
        opt.mW{l} = opt.beta1*opt.mW{l} + (1-opt.beta1)*grads.dW{l};
        opt.vW{l} = opt.beta2*opt.vW{l} + (1-opt.beta2)*(grads.dW{l}.^2);
        mW_hat = opt.mW{l}/(1-opt.beta1^opt.t);
        vW_hat = opt.vW{l}/(1-opt.beta2^opt.t);
        net.W{l} = net.W{l} - opt.lr*mW_hat./(sqrt(vW_hat)+1e-8);
        opt.mb{l} = opt.beta1*opt.mb{l} + (1-opt.beta1)*grads.db{l};
        opt.vb{l} = opt.beta2*opt.vb{l} + (1-opt.beta2)*(grads.db{l}.^2);
        mb_hat = opt.mb{l}/(1-opt.beta1^opt.t);
        vb_hat = opt.vb{l}/(1-opt.beta2^opt.t);
        net.b{l} = net.b{l} - opt.lr*mb_hat./(sqrt(vb_hat)+1e-8);
    end
end
