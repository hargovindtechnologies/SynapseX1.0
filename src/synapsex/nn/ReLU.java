/**
 * Copyright (c) 2025 Hargovind Technologies. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-Hargovind-1.0
 *
 * See the LICENSE file in the project root for license terms.
 *
 */
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
