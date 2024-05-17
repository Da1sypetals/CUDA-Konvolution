#include <cstdio>

#define THREADS_PER_BLOCK 256
#define WARP_SIZE 32
#define DIVUP(m, n) ((m + n - 1) / n)
#define INDEX2D(a, b, db) (((a) * (db) + (b)))
#define INDEX3D(a, b, c, db, dc) (((a) * (db) * (dc) + (b) * (dc) + (c)))

__global__ void leg2d_fwd_kernel(const float *x, float *leg, 
                    int batch_size, int in_channels, int height, int width, int degree, int numThreads){
                        
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numThreads) {

        // int BC = batch_size * in_channels;
        int HW = height * width;

        int bc = idx / HW;
        int I = idx % HW;
        
        float x_val = x[idx];
        leg[INDEX3D(bc, 1, I, degree + 1, HW)] = x_val;
        float leg_val_z = x_val; // index = i - 1
        float leg_val_zz = 1;    // index = i - 2

        for(int d = 2; d < degree + 1; d++){
            float df = static_cast<float>(d);
            float denom_inv = 1.f / df;
            float new_leg_val = ((2 * df - 1) * x_val * leg_val_z - (df - 1) * x_val * leg_val_zz) * denom_inv;
            leg[INDEX3D(bc, d, I, degree + 1, HW)] = new_leg_val;


            // finally
            leg_val_zz = leg_val_z;
            leg_val_z = new_leg_val;
        }
    }
}


__global__ void leg2d_bwd_kernel(const float* gout, const float *x, const float *leg, float* grad_x,
                                 int batch_size, int in_channels, int height, int width, int degree, int numThreads){

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numThreads) {

        int HW = height * width;

        int bc = idx / HW;
        int I = idx % HW;

        float b0 = 0, b1 = 1;
        float b_z = b1, b_zz = b0; // b(i-1) and b(i-2)

        float x_val = x[idx];

        // grad wrt d=0 is equal to zero
        // here is grad wrt d=1
        float grad_x_val = gout[INDEX3D(bc, 1, I, degree + 1, HW)];

        for(int d = 2; d < degree + 1; d++){

            float df = static_cast<float>(d);
            float denom_inv = 1.f / df;

            // 2a(i-1)
            float b = ((2 * df - 1) * (leg[INDEX3D(bc, d - 1, I, degree + 1, HW)] + x_val * b_z) - (df - 1) * b_zz) * denom_inv;

            grad_x_val += gout[INDEX3D(bc, d, I, degree + 1, HW)] * b;

            // finally
            b_zz = b_z;
            b_z = b;
        }

        grad_x[idx] = grad_x_val;


    }
}



void leg2d_launcher(const float *x, float *leg,
                    int batch_size, int in_channels, int height, int width, int degree){
                        
    int numThreads = batch_size * (in_channels * width * height);
    dim3 blockSize(DIVUP(numThreads, THREADS_PER_BLOCK));
    dim3 threadSize(THREADS_PER_BLOCK);
    leg2d_fwd_kernel<<<blockSize, threadSize>>>(x, leg, batch_size, in_channels, height, width, degree, numThreads);
}

void leg2d_bwd_launcher(const float *gout, const float *x, const float *leg, float *grad_x,
                        int batch_size, int in_channels, int width, int height, int degree){

    int numThreads = batch_size * (in_channels * width * height);
    dim3 blockSize(DIVUP(numThreads, THREADS_PER_BLOCK));
    dim3 threadSize(THREADS_PER_BLOCK);
    leg2d_bwd_kernel<<<blockSize, threadSize>>>(gout, x, leg, grad_x, batch_size, in_channels, height, width, degree, numThreads);
}


