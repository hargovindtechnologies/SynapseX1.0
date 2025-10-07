/**
 * Copyright (c) 2025 Hargovind Technologies. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-Hargovind-1.0
 *
 * See the LICENSE file in the project root for license terms.
 *
 */
package synapsex.nn;

import synapsex.core.Tensor;

import java.util.Arrays;
import java.util.Random;

/**
 * Simple Linear layer y = x @ W^T + b
 * Accepts input x with shape [batch, inFeatures]
 * weight shape = [outFeatures, inFeatures]
 * bias shape = [1, outFeatures] (broadcasted)
 * @author Hargovind Singh
 */
public class Linear extends Module {
    public final int inFeatures;
    public final int outFeatures;
    public final Tensor weight;
    public final Tensor bias;

    public Linear(int inFeatures, int outFeatures) {
        this.inFeatures = inFeatures;
        this.outFeatures = outFeatures;
        this.weight = Tensor.randn(outFeatures, inFeatures).setRequiresGrad(true);
        this.bias = Tensor.zeros(1, outFeatures);
        // simple init
        Random r = new Random();
        for (int i = 0; i < weight.size; i++) weight.data[i] = r.nextGaussian() * Math.sqrt(2.0 / inFeatures);
        for (int i = 0; i < bias.size; i++) bias.data[i] = 0.0;
        bias.setRequiresGrad(true);
        registerParam("weight", weight);
        registerParam("bias", bias);
    }

    @Override
    public Tensor forward(Tensor x) {
        // x: [batch, in], weight: [out, in] -> out = x @ weight^T + bias
        if (x.shape.length != 2) throw new IllegalArgumentException("Linear expects 2D input");
        int batch = x.shape[0];
        int in = x.shape[1];
        if (in != inFeatures) throw new IllegalArgumentException("input dim mismatch");

        Tensor out = new Tensor(batch, outFeatures);
        for (int b = 0; b < batch; b++) {
            for (int j = 0; j < outFeatures; j++) {
                double s = 0.0;
                for (int k = 0; k < in; k++) s += x.data[b*in + k] * weight.data[j*in + k];
                out.data[b*outFeatures + j] = s + bias.data[j];
            }
        }

        // autograd
        if (x.requiresGrad || weight.requiresGrad || bias.requiresGrad) {
            out.setRequiresGrad(true);
            out.parents.add(x);
            out.parents.add(weight);
            out.parents.add(bias);
            out.gradFn = (self, up) -> {
                // up shape [batch, outFeatures]
                if (x.requiresGrad) {
                    Arrays.fill(x.grad, 0.0);
                    for (int b = 0; b < batch; b++) {
                        for (int i = 0; i < in; i++) {
                            double s = 0.0;
                            for (int j = 0; j < outFeatures; j++) s += up[b*outFeatures + j] * weight.data[j*in + i];
                            x.grad[b*in + i] += s;
                        }
                    }
                }
                if (weight.requiresGrad) {
                    Arrays.fill(weight.grad, 0.0);
                    for (int j = 0; j < outFeatures; j++) {
                        for (int k = 0; k < in; k++) {
                            double s = 0.0;
                            for (int b = 0; b < batch; b++) s += up[b*outFeatures + j] * x.data[b*in + k];
                            weight.grad[j*in + k] += s;
                        }
                    }
                }
                if (bias.requiresGrad) {
                    Arrays.fill(bias.grad, 0.0);
                    for (int j = 0; j < outFeatures; j++) {
                        double s = 0.0;
                        for (int b = 0; b < batch; b++) s += up[b*outFeatures + j];
                        bias.grad[j] += s;
                    }
                }
            };
        }
        return out;
    }
}
