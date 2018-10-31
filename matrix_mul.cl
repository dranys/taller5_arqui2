kernel void matrix_multiplication(__global float *A,__global float *B, __global float *C) {

   float sum;//final result
   int i = get_global_size(0);//get element to process
   int Vectors_Rows = i/4;//calculate row vector
   int init = get_global_id(0) * Vectors_Rows;//calc init
   A += init;
   C += init*4;
   int j;
   int k;
   for( j=0; j<i; j++) {// go throught each column of matrix
      sum = 0.0f;
      for(int k=0; k<Vectors_Rows; ++) {//  go throught each row of matrix
         sum += dot(A[j],
                B[i*Vectors_Rows + k]);
      }
      C[i] = sum;
      }
}
