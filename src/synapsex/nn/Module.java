package synapsex.nn;

import synapsex.core.Tensor;
import java.util.*;

/**
 * Base Module: holds named parameters and requires subclass to implement forward()
 * @author Hargovind Singh
 */
public abstract class Module {
    protected final LinkedHashMap<String, Tensor> params = new LinkedHashMap<>();

    public abstract Tensor forward(Tensor x);

    public List<Tensor> parameters() {
        return new ArrayList<>(params.values());
    }

    public void zeroGrad() {
        for (Tensor t : parameters()) t.zeroGrad();
    }

    protected void registerParam(String name, Tensor t) {
        params.put(name, t);
    }
}
