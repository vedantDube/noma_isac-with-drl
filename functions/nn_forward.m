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
