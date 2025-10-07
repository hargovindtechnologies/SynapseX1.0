package synapsex.nn;

import synapsex.core.Tensor;
import java.util.*;

/** Container module that runs modules sequentially
 * @author Hargovind Singh
 * */
public class Sequential extends Module {
    private final List<Module> modules = new ArrayList<>();

    public Sequential(Module... mods) {
        for (Module m : mods) modules.add(m);
    }

    @Override
    public Tensor forward(Tensor x) {
        Tensor t = x;
        for (Module m : modules) t = m.forward(t);
        return t;
    }

    @Override
    public List<Tensor> parameters() {
        List<Tensor> out = new ArrayList<>();
        for (Module m : modules) out.addAll(m.parameters());
        return out;
    }
}
