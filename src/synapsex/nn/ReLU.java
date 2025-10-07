package synapsex.nn;

import synapsex.core.Tensor;

/** Stateless ReLU Module
 * @author Hargovind Singh
 * */
public class ReLU extends Module {
    @Override
    public Tensor forward(Tensor x) {
        return Tensor.relu(x);
    }
}
