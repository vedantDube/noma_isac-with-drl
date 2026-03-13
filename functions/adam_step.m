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
