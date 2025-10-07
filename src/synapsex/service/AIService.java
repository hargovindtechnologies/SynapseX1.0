package com.synapsex.service;

import org.springframework.stereotype.Service;
import synapsex.core.Tensor;
import synapsex.nn.Linear;
import synapsex.nn.ReLU;
import synapsex.nn.Sequential;

@Service
public class AIService {

    private final Sequential model;

    public AIService() {
        // initialize a small model
        Linear l1 = new Linear(4, 16);
        ReLU r = new ReLU();
        Linear l2 = new Linear(16, 2);
        model = new Sequential(l1, r, l2);
    }

    /**
     * Make prediction
     * @param input double[][] array shape [batch, features]
     * @return output double[][] array
     */
    public double[][] predict(double[][] input) {
        int batch = input.length;
        int features = input[0].length;
        Tensor x = new Tensor(batch, features);
        for (int i = 0; i < batch; i++) {
            System.arraycopy(input[i], 0, x.data, i*features, features);
        }
        Tensor out = model.forward(x);
        double[][] result = new double[batch][out.shape[1]];
        for (int i = 0; i < batch; i++) {
            for (int j = 0; j < out.shape[1]; j++) {
                result[i][j] = out.data[i*out.shape[1] + j];
            }
        }
        return result;
    }
}
