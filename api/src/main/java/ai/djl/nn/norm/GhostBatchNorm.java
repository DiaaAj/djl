package ai.djl.nn.norm;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.training.ParameterStore;
import ai.djl.translate.Batchifier;
import ai.djl.translate.StackBatchifier;
import ai.djl.util.PairList;

/**
 * {@code GhostBatchNorm} is similar to {@code BatchNorm} except that it splits a batch into a
 * smaller sub-batches aka <em>ghost batches</em>, and normalize them individually to have a mean of
 * 0 and variance of 1 and finally concatenate them again to a single batch. Each of the
 * mini-batches contains a {@code virtualBatchSize} samples.
 */
public class GhostBatchNorm extends BatchNorm {

    private int virtualBatchSize;
    private Batchifier batchifier;

    GhostBatchNorm(Builder builder) {
        super(new BatchNorm.Builder());

        this.virtualBatchSize = builder.virtualBatchSize;
        this.batchifier = new StackBatchifier();
    }

    /** {@inheritDoc} */
    @Override
    protected NDList forwardInternal(
            ParameterStore parameterStore,
            NDList inputs,
            boolean training,
            PairList<String, Object> params) {

        NDList[] subBatches = split(inputs);
        for (NDList batch : subBatches)
            super.forwardInternal(parameterStore, batch, training, params);

        return batchifier.batchify(subBatches);
    }

    /**
     * Set the size of virtual batches in which to use when sub-batching. Defaults to 128.
     *
     * @param virtualBatchSize
     * @return
     */
    public void optVirtualBatchSize(int virtualBatchSize) {
        this.virtualBatchSize = virtualBatchSize;
    }

    /**
     * Splits an {@code NDArray} into the given <b> size </b> of sub-batch. If the batch size is
     * divisible by the virtual batch size, all returned sub-batches will be the same size. If the
     * batch size is not divisible by virtual batch size, all returned sub-batches will be the same
     * size, except the last one.
     *
     * @param list the {@link NDList} that needs to be split
     * @return
     */
    protected NDList[] split(NDList list) {
        double batchSize = list.head().size(0);
        int countBatches = (int) Math.ceil(batchSize / virtualBatchSize);

        return batchifier.split(list, countBatches, true);
    }

    /**
     * Creates a builder to build a {@code GhostBatchNorm.}
     *
     * @return
     */
    public static Builder builder() {
        return new Builder();
    }

    /** The Builder to construct a {@link GhostBatchNorm} */
    public static class Builder extends BatchNorm.Builder {
        private int virtualBatchSize = 128;

        Builder() {}

        /** {@inheritDoc} */
        public Builder optVirtualBatchSize(int virtualBatchSize) {
            this.virtualBatchSize = virtualBatchSize;
            return this;
        }

        public GhostBatchNorm build() {
            return new GhostBatchNorm(this);
        }

        /** {@inheritDoc} */
        @Override
        public Builder optAxis(int axis) {
            super.optAxis(axis);
            return this;
        }

        /** {@inheritDoc} */
        @Override
        public Builder optCenter(boolean val) {
            super.optCenter(val);
            return this;
        }

        /** {@inheritDoc} */
        @Override
        public Builder optScale(boolean val) {
            super.optScale(val);
            return this;
        }

        /** {@inheritDoc} */
        @Override
        public Builder optEpsilon(float val) {
            super.optEpsilon(val);
            return this;
        }

        /** {@inheritDoc} */
        @Override
        public Builder optMomentum(float val) {
            super.optMomentum(val);
            return this;
        }
    }
}
