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
