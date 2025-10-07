/**
 * Copyright (c) 2025 Hargovind Technologies. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-Hargovind-1.0
 *
 * See the LICENSE file in the project root for license terms.
 *
 */
import synapsex.core.Tensor;
import synapsex.nn.*;
import synapsex.nn.loss.MSELoss;
import synapsex.optim.SGD;
import java.util.*;

public class ExampleMain {
    public static void main(String[] args) {
        int batch = 8;
        int input = 4;
        int hidden = 16;
        int out = 2;
        int epochs = 100;

        // model
        Linear l1 = new Linear(input, hidden);
        ReLU r = new ReLU();
        Linear l2 = new Linear(hidden, out);
        Sequential model = new Sequential(l1, r, l2);

        // enable grads for parameters already done in Linear constructor
        List<synapsex.core.Tensor> params = model.parameters();

        SGD opt = new SGD(params, 0.05);

        // toy dataset: learn to map random x -> linear target y = A x + b (so model should fit)
        Random rnd = new Random(42);
        double[] A = new double[input * out];
        double[] B = new double[out];
        for (int i = 0; i < A.length; i++) A[i] = rnd.nextGaussian();
        for (int i = 0; i < B.length; i++) B[i] = rnd.nextGaussian();

        for (int e = 0; e < epochs; e++) {
            // batch inputs
            Tensor x = new Tensor(batch, input);
            Tensor y = new Tensor(batch, out);
            for (int b = 0; b < batch; b++) {
                for (int i = 0; i < input; i++) x.data[b*input + i] = rnd.nextGaussian();
                // targets: linear transform
                for (int j = 0; j < out; j++) {
                    double s = 0.0;
                    for (int i = 0; i < input; i++) s += x.data[b*input + i] * A[i*out + j];
                    y.data[b*out + j] = s + B[j];
                }
            }
            // forward
            Tensor preds = model.forward(x);
            // compute loss
            Tensor loss = MSELoss.mse(preds, y);
            // zero grads
            model.zeroGrad();
            // backward (scalar)
            loss.backward();
            // step
            opt.step();
            if (e % 10 == 0) System.out.printf("Epoch %d loss=%.6f%n", e, loss.data[0]);
        }
        System.out.println("Training done.");
    }
}
