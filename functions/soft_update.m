function tgt = soft_update(net, tgt, tau)
    for l = 1:net.L
        tgt.W{l} = tau * net.W{l} + (1 - tau) * tgt.W{l};
        tgt.b{l} = tau * net.b{l} + (1 - tau) * tgt.b{l};
    end
end
