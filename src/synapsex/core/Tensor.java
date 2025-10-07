/**
 * Copyright (c) 2025 Hargovind Technologies. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-Hargovind-1.0
 *
 * See the LICENSE file in the project root for license terms.
 *
 */
package synapsex.core;

import java.util.*;
import java.util.function.BiConsumer;

/**
 * Minimal Tensor with:
 * - row-major flattened double[] storage
 * - arbitrary shape (1D/2D common cases)
 * - very small autograd graph (parents + gradFn)
 *
 * Limitations: not optimized, backward supports scalar root only,
 * broadcasting is limited (not implemented fully).
 * @author Hargovind Singh
 */
public class Tensor {
    public final double[] data;
    public final int[] shape;
    public final int size;

    // autograd
    public double[] grad;                // same length as data
    public boolean requiresGrad = false;
    public final List<Tensor> parents = new ArrayList<>();
    public BiConsumer<Tensor,double[]> gradFn = null; // (self, upstreamGrad)

    // constructors
    public Tensor(int... shape) {
        this.shape = shape.clone();
        this.size = computeSize(shape);
        this.data = new double[this.size];
        this.grad = new double[this.size];
    }

    public Tensor(double[] data, int... shape) {
        this.shape = shape.clone();
        this.size = computeSize(shape);
        if (data.length != this.size) throw new IllegalArgumentException("data length mismatch");
        this.data = data.clone();
        this.grad = new double[this.size];
    }

    private static int computeSize(int[] shape) {
        int s = 1;
        for (int d : shape) s *= d;
        return s;
    }

    // factory helpers
    public static Tensor zeros(int... shape) { return new Tensor(shape); }

    public static Tensor randn(int... shape) {
        Tensor t = new Tensor(shape);
        Random r = new Random();
        for (int i = 0; i < t.size; i++) t.data[i] = r.nextGaussian() * 0.01;
        return t;
    }

    public static Tensor fromScalar(double v) {
        return new Tensor(new double[]{v}, 1);
    }

    // enable grad
    public Tensor setRequiresGrad(boolean flag) {
        this.requiresGrad = flag;
        return this;
    }

    // utility
    public String shapeString() { return Arrays.toString(shape); }

    public void zeroGrad() { Arrays.fill(this.grad, 0.0); }

    // ========== Basic Ops (no broadcasting) ==========
    public static Tensor add(Tensor a, Tensor b) {
        if (!Arrays.equals(a.shape, b.shape)) throw new IllegalArgumentException("shape mismatch for add");
        Tensor out = new Tensor(a.shape);
        for (int i = 0; i < a.size; i++) out.data[i] = a.data[i] + b.data[i];

        if (a.requiresGrad || b.requiresGrad) {
            out.setRequiresGrad(true);
            out.parents.add(a);
            out.parents.add(b);
            out.gradFn = (self, up) -> {
                if (a.requiresGrad) for (int i = 0; i < a.size; i++) a.grad[i] += up[i];
                if (b.requiresGrad) for (int i = 0; i < b.size; i++) b.grad[i] += up[i];
            };
        }
        return out;
    }

    public static Tensor sub(Tensor a, Tensor b) {
        if (!Arrays.equals(a.shape, b.shape)) throw new IllegalArgumentException("shape mismatch for sub");
        Tensor out = new Tensor(a.shape);
        for (int i = 0; i < a.size; i++) out.data[i] = a.data[i] - b.data[i];

        if (a.requiresGrad || b.requiresGrad) {
            out.setRequiresGrad(true);
            out.parents.add(a);
            out.parents.add(b);
            out.gradFn = (self, up) -> {
                if (a.requiresGrad) for (int i = 0; i < a.size; i++) a.grad[i] += up[i];
                if (b.requiresGrad) for (int i = 0; i < b.size; i++) b.grad[i] -= up[i];
            };
        }
        return out;
    }

    public static Tensor mul(Tensor a, Tensor b) {
        if (!Arrays.equals(a.shape, b.shape)) throw new IllegalArgumentException("shape mismatch for mul");
        Tensor out = new Tensor(a.shape);
        for (int i = 0; i < a.size; i++) out.data[i] = a.data[i] * b.data[i];

        if (a.requiresGrad || b.requiresGrad) {
            out.setRequiresGrad(true);
            out.parents.add(a);
            out.parents.add(b);
            out.gradFn = (self, up) -> {
                if (a.requiresGrad) for (int i = 0; i < a.size; i++) a.grad[i] += up[i] * b.data[i];
                if (b.requiresGrad) for (int i = 0; i < b.size; i++) b.grad[i] += up[i] * a.data[i];
            };
        }
        return out;
    }

    // matrix multiplication for 2D tensors only (shape: [m,k] x [k,n] -> [m,n])
    public static Tensor matmul(Tensor A, Tensor B) {
        if (A.shape.length != 2 || B.shape.length != 2) throw new IllegalArgumentException("matmul expects 2D tensors");
        int m = A.shape[0], k = A.shape[1], k2 = B.shape[0], n = B.shape[1];
        if (k != k2) throw new IllegalArgumentException("matmul inner dim mismatch");
        Tensor out = new Tensor(m, n);
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                double s = 0.0;
                for (int t = 0; t < k; t++) s += A.data[i*k + t] * B.data[t*n + j];
                out.data[i*n + j] = s;
            }
        }

        if (A.requiresGrad || B.requiresGrad) {
            out.setRequiresGrad(true);
            out.parents.add(A);
            out.parents.add(B);
            out.gradFn = (self, up) -> {
                // up is shape [m,n]
                if (A.requiresGrad) {
                    Arrays.fill(A.grad, 0.0);
                    for (int i = 0; i < m; i++) {
                        for (int t = 0; t < k; t++) {
                            double s = 0.0;
                            for (int j = 0; j < n; j++) s += up[i*n + j] * B.data[t*n + j];
                            A.grad[i*k + t] += s;
                        }
                    }
                }
                if (B.requiresGrad) {
                    Arrays.fill(B.grad, 0.0);
                    for (int t = 0; t < k; t++) {
                        for (int j = 0; j < n; j++) {
                            double s = 0.0;
                            for (int i = 0; i < m; i++) s += A.data[i*k + t] * up[i*n + j];
                            B.grad[t*n + j] += s;
                        }
                    }
                }
            };
        }
        return out;
    }

    // elementwise ReLU
    public static Tensor relu(Tensor a) {
        Tensor out = new Tensor(a.shape);
        for (int i = 0; i < a.size; i++) out.data[i] = Math.max(0.0, a.data[i]);

        if (a.requiresGrad) {
            out.setRequiresGrad(true);
            out.parents.add(a);
            out.gradFn = (self, up) -> {
                for (int i = 0; i < a.size; i++) {
                    if (a.data[i] > 0) a.grad[i] += up[i];
                }
            };
        }
        return out;
    }

    // sum to scalar
    public Tensor sum() {
        double s = 0.0;
        for (double v : data) s += v;
        Tensor out = fromScalar(s);
        if (this.requiresGrad) {
            out.setRequiresGrad(true);
            out.parents.add(this);
            out.gradFn = (self, up) -> {
                for (int i = 0; i < this.size; i++) this.grad[i] += up[0];
            };
        }
        return out;
    }

    // mean -> scalar
    public Tensor mean() {
        Tensor s = this.sum();
        double factor = 1.0 / this.size;
        s.data[0] *= factor;
        if (s.requiresGrad) {
            BiConsumer<Tensor,double[]> old = s.gradFn;
            s.gradFn = (self, up) -> {
                double scaled = up[0] * factor;
                old.accept(self, new double[]{scaled});
            };
        }
        return s;
    }

    // ========== Autograd backward (scalar root) ==========
    public void backward() {
        if (this.size != 1) throw new IllegalStateException("backward() expects a scalar (size==1) as root");
        // initialize grads
        Arrays.fill(this.grad, 0.0);
        this.grad[0] = 1.0;
        // topo sort
        List<Tensor> topo = new ArrayList<>();
        Set<Tensor> seen = new HashSet<>();
        buildTopo(this, topo, seen);
        // reverse traversal: call gradFn on each
        for (int i = topo.size()-1; i >= 0; i--) {
            Tensor t = topo.get(i);
            if (t.gradFn != null) {
                t.gradFn.accept(t, t.grad);
            }
        }
    }

    private static void buildTopo(Tensor t, List<Tensor> topo, Set<Tensor> seen) {
        if (seen.contains(t)) return;
        seen.add(t);
        for (Tensor p : t.parents) buildTopo(p, topo, seen);
        topo.add(t);
    }

    // debugging print
    public void print() {
        System.out.println("Tensor(shape=" + Arrays.toString(shape) + ", data=" + Arrays.toString(data) + ")");
    }
}
