package synapsex.nn.loss;

import synapsex.core.Tensor;

/**
 * Mean Squared Error producing a scalar Tensor (mean over all elements).
 * Usage: Tensor loss = MSELoss.mse(preds, targets);
 * then loss.backward();
 * @author Hargovind Singh
 */
public class MSELoss {
    public static Tensor mse(Tensor preds, Tensor targets) {
        if (!java.util.Arrays.equals(preds.shape, targets.shape)) throw new IllegalArgumentException("shape mismatch");
        // diff = preds - targets
        Tensor diff = Tensor.sub(preds, targets);
        // sq = diff * diff
        Tensor sq = Tensor.mul(diff, diff);
        // mean -> scalar
        return sq.mean();
    }
}
