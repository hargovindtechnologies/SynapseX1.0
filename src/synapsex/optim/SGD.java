package synapsex.optim;

import synapsex.core.Tensor;
import java.util.*;

/** Very small SGD optimizer (no momentum, no weight decay)
 * @author Hargovind Singh
 * */
public class SGD {
    private final List<Tensor> params;
    private final double lr;

    public SGD(List<Tensor> params, double lr) {
        this.params = params;
        this.lr = lr;
    }

    public void step() {
        for (Tensor p : params) {
            for (int i = 0; i < p.size; i++) {
                p.data[i] -= lr * p.grad[i];
            }
        }
    }

    public void zeroGrad() {
        for (Tensor p : params) Arrays.fill(p.grad, 0.0);
    }
}
