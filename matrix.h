#ifndef MATRIX_H
#define MATRIX_H

typedef struct Matrix {
    int rows;
    int columns;
    double **arr;
} Matrix;

Matrix *matrix_create(int m, int n);
void matrix_destroy(Matrix *mtrx);
void matrix_set_entry(Matrix *mtrx, int i, int j, double n);
double matrix_get_entry(Matrix *mtrx, int i, int j);
double *matrix_get_row(Matrix *mtrx, int i);
double *matrix_get_col(Matrix *mtrx, int j);
double dot_product(double *a0, double *a1, int n);
Matrix *matrix_add(Matrix *a, Matrix *b);
Matrix *matrix_scalar_multiply(Matrix *a, double c);
double matrix_ij_cofactor(Matrix *mtrx, int i, int j);
Matrix *matrix_ij(Matrix *mtrx, int i, int j);
Matrix *matrix_transpose(Matrix *mtrx);
double matrix_determinant(Matrix *mtrx);
Matrix *matrix_adjugate(Matrix *mtrx);
Matrix *matrix_inverse(Matrix *mtrx);
void matrix_print(Matrix *mtrx);
Matrix *matrix_cofactor(Matrix *mtrx);
Matrix *matrix_power(Matrix *mtrx, int n);


#endif