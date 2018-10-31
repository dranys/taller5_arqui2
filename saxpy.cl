__kernel void saxpy(__global const long *A, __global const long *B, __global long *C,  __global const long *saxpyNumber) {
 
    // Get the index of the current element to be processed
    int i = get_global_id(0);
 
    // Do the operation
    C[i] = (* saxpyNumber) * A[i] + B[i];
}

