#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;

class ReverseItOp : public OpKernel {
public:
    explicit ReverseItOp(OpKernelConstruction* context) : OpKernel(context) {}

    void Compute(OpKernelContext* context) override {
        // Grab the input tensor
        const Tensor& input_tensor = context->input(0);
        auto input = input_tensor.flat<int32>();

        // Create an output tensor
        Tensor* output_tensor = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(),
            &output_tensor));
        auto output_flat = output_tensor->flat<int32>();

        // Reverse the tensor
        const int N = input.size();
        for (int i = 0; i < N; i++) {
            output_flat(i) = input(N - i - 1);
        }
    }
};

REGISTER_OP("ReverseIt")
    .Input("to_zero: int32")
    .Output("zeroed: int32")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        c->set_output(0, c->input(0));
        return Status::OK();
    });

REGISTER_KERNEL_BUILDER(Name("ReverseIt").Device(DEVICE_CPU), ReverseItOp);

