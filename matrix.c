#include "matrix.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>

Matrix *matrix_create(int m, int n) {
    Matrix *new = malloc(sizeof(Matrix));
    new->rows = m;
    new->columns = n;
    new->arr = malloc(m * sizeof(double *));
    for (int i = 0; i < m; i++) {
        new->arr[i] = calloc(n, sizeof(double));
    }
    return new;
}

void matrix_destroy(Matrix *mtrx) {
    for (int i = 0; i < mtrx->rows; i++) {
        free(mtrx->arr[i]);
    }
    free(mtrx->arr);
    free(mtrx);
    return;
}

void matrix_set_entry(Matrix *mtrx, int i, int j, double n) {
    mtrx->arr[i-1][j-1] = n;
    return;
}

double matrix_get_entry(Matrix *mtrx, int i, int j) {
    return mtrx->arr[i-1][j-1];
}

double *matrix_get_row(Matrix *mtrx, int i) {
    return mtrx->arr[i-1];
}

//Must free the returned array after use.
double *matrix_get_col(Matrix *mtrx, int j) {
    double *col = malloc((mtrx->rows) * sizeof(double));
    for (int i = 0; i < mtrx->rows; i++) {
        col[i] = mtrx->arr[i][j-1];
    }
    return col;
}

double dot_product(double *a0, double *a1, int n) {
    double res = 0;
    for (int i = 0; i < n; i++) {
        res += a0[i] * a1[i];
    }
    return res;
}

Matrix *matrix_add(Matrix *a, Matrix *b) {
    int m = a->rows;
    int n = a->columns;
    for (int i = 1; i <= m; i++) {
        for (int j = 1; j <= n; j++) {
            double aij = matrix_get_entry(a, i, j);
            double bij = matrix_get_entry(b, i, j);
            matrix_set_entry(a, i, j, aij + bij);
        }
    }
    return a;
}

Matrix *matrix_scalar_multiply(Matrix *mtrx, double c) {
    int m = mtrx->rows;
    int n = mtrx->columns;
    for (int i = 1; i <= m; i++) {
        for (int j = 1; j <= n; j++) {
            double entry = matrix_get_entry(mtrx, i, j);
            matrix_set_entry(mtrx, i, j, c * entry);
        }
    }
    return mtrx;
}

Matrix *matrix_multiply(Matrix *a, Matrix *b) {
    int m = a->rows;
    int k = b->columns;
    Matrix *res = matrix_create(m, k);
    for (int i = 1; i <= m; i++) {
        for (int j = 1; j <= k; j++) {
            double *ithrow = matrix_get_row(a, i);
            double *jthcol = matrix_get_col(b, j);
            double dotprod = dot_product(ithrow, jthcol, a->columns);
            matrix_set_entry(res, i, j, dotprod);
            free(jthcol);
        }
    }
    return res;
}

void matrix_print(Matrix *mtrx) {
    for (int i = 1; i <= mtrx->rows; i++) {
        for (int j = 1; j <= mtrx->columns; j++) {
            printf("%g ", matrix_get_entry(mtrx, i, j));
        }
        printf("\n");
    }
    return;
}

Matrix *matrix_ij(Matrix *mtrx, int i, int j) {
    int m = mtrx->rows - 1;
    int n = mtrx->columns - 1;
    Matrix *res = matrix_create(m, n);
    for (int k = 1; k <= m; k++) {
        int adjk = (k >= i ? k + 1 : k);
        for (int r = 1; r <= n; r++) {
            int adjr = (r >= j ? r + 1 : r);
            double entry = matrix_get_entry(mtrx, adjk, adjr);
            matrix_set_entry(res, k, r, entry);
        }
    }
    return res;
}

Matrix *matrix_transpose(Matrix *mtrx) {
    int m = mtrx->rows;
    int n = mtrx->columns;
    Matrix *res = matrix_create(n, m);
    for (int i = 1; i <= n; i++) {
        for (int j = 1; j <= m; j++) {
            double entry = matrix_get_entry(mtrx, j, i);
            matrix_set_entry(res, i, j, entry);
        }
    }
    return res;
}

double matrix_ij_cofactor(Matrix *mtrx, int i, int j) {
    Matrix *ij = matrix_ij(mtrx, i, j);
    double res = pow(-1, i + j) * matrix_determinant(ij);
    matrix_destroy(ij);
    return res;
}

double matrix_determinant(Matrix *mtrx) {
    if (mtrx->rows == 2) {
        double a = matrix_get_entry(mtrx, 1, 1);
        double b = matrix_get_entry(mtrx, 1, 2);
        double c = matrix_get_entry(mtrx, 2, 1);
        double d = matrix_get_entry(mtrx, 2, 2);
        return a*d - b*c;
    }
    double res = 0;
    for (int j = 1; j <= mtrx->columns; j++) {
        res += matrix_get_entry(mtrx, 1, j) * matrix_ij_cofactor(mtrx, 1, j);
    }
    return res;
}

Matrix *matrix_cofactor(Matrix *mtrx) {
    int m = mtrx->rows;
    int n = mtrx->columns;
    Matrix *res = matrix_create(m, n);
    for (int i = 1; i <= m; i++) {
        for (int j = 1; j <= n; j++) {
            double ij_cofactor = matrix_ij_cofactor(mtrx, i, j);
            matrix_set_entry(res, i, j, ij_cofactor);
        }
    }
    return res;
}

Matrix *matrix_adjugate(Matrix *mtrx) {
    if (mtrx->rows == 2) {
        Matrix *adj = matrix_create(2, 2);
        double a = matrix_get_entry(mtrx, 1, 1);
        double b = matrix_get_entry(mtrx, 1, 2);
        double c = matrix_get_entry(mtrx, 2, 1);
        double d = matrix_get_entry(mtrx, 2, 2);
        matrix_set_entry(adj, 1, 1, d);
        matrix_set_entry(adj, 1, 2, -b);
        matrix_set_entry(adj, 2, 1, -c);
        matrix_set_entry(adj, 2, 2, a);
        return adj;
    }
    Matrix *cof = matrix_cofactor(mtrx);
    Matrix *adj = matrix_transpose(cof);
    matrix_destroy(cof);
    return adj;
}

Matrix *matrix_inverse(Matrix *mtrx) {
    double det = matrix_determinant(mtrx);
    if (det == 0) {
        printf("Matrix not invertible.\n");
        return NULL;
    } else {
        Matrix *adj = matrix_adjugate(mtrx);
        matrix_scalar_multiply(adj, 1/det);
        return adj;
    }
}

Matrix *matrix_power(Matrix *mtrx, int n) {
    Matrix *res = matrix_multiply(mtrx, mtrx);
    for (int i = n; i > 2; i--) {
        Matrix *new = matrix_multiply(res, mtrx);
        matrix_destroy(res);
        res = new;
    }
    return res;
}

int main(void) {
    printf("==Matrices==\n\n");

    Matrix *mtrx1 = matrix_create(2, 2);
    matrix_set_entry(mtrx1, 1, 1, 2);
    matrix_set_entry(mtrx1, 1, 2, 1);
    matrix_set_entry(mtrx1, 2, 1, -1);
    matrix_set_entry(mtrx1, 2, 2, 3);
    assert(matrix_get_entry(mtrx1, 1, 1) == 2);
    assert(matrix_get_entry(mtrx1, 2, 2) == 3);
    matrix_print(mtrx1);
    printf("\n");

    Matrix *mtrx2 = matrix_create(3, 4);
    matrix_set_entry(mtrx2, 1, 1, 1);
    matrix_set_entry(mtrx2, 1, 2, 2);
    matrix_set_entry(mtrx2, 1, 3, 3);
    matrix_set_entry(mtrx2, 1, 4, 4);
    matrix_set_entry(mtrx2, 2, 1, 0);
    matrix_set_entry(mtrx2, 2, 2, -1);
    matrix_set_entry(mtrx2, 2, 3, 4);
    matrix_set_entry(mtrx2, 2, 4, 9);
    matrix_set_entry(mtrx2, 3, 1, 0);
    matrix_set_entry(mtrx2, 3, 2, 0);
    matrix_set_entry(mtrx2, 3, 3, 3);
    matrix_set_entry(mtrx2, 3, 4, 2);
    assert(matrix_get_entry(mtrx2, 3, 2) == 0);
    assert(matrix_get_entry(mtrx2, 1, 3) == 3);
    assert(matrix_get_entry(mtrx2, 2, 4) == 9);
    matrix_print(mtrx2);
    printf("\n");

    Matrix *mtrx3 = matrix_create(4, 2);
    matrix_set_entry(mtrx3, 1, 1, 0);
    matrix_set_entry(mtrx3, 1, 2, 2);
    matrix_set_entry(mtrx3, 2, 1, 4);
    matrix_set_entry(mtrx3, 2, 2, 5);
    matrix_set_entry(mtrx3, 3, 1, 1);
    matrix_set_entry(mtrx3, 3, 2, 2);
    matrix_set_entry(mtrx3, 4, 1, 0);
    matrix_set_entry(mtrx3, 4, 2, 0);
    assert(matrix_get_entry(mtrx3, 1, 2) == 2);
    assert(matrix_get_entry(mtrx3, 3, 1) == 1);
    assert(matrix_get_entry(mtrx3, 4, 2) == 0);
    matrix_print(mtrx3);
    printf("\n");

    Matrix *mtrx4 = matrix_create(2, 3);
    matrix_set_entry(mtrx4, 1, 1, 1);
    matrix_set_entry(mtrx4, 1, 2, 5);
    matrix_set_entry(mtrx4, 1, 3, -2);
    matrix_set_entry(mtrx4, 2, 1, 2);
    matrix_set_entry(mtrx4, 2, 2, 6);
    matrix_set_entry(mtrx4, 2, 3, -3);
    assert(matrix_get_entry(mtrx4, 1, 1) == 1);
    assert(matrix_get_entry(mtrx4, 1, 3) == -2);
    assert(matrix_get_entry(mtrx4, 2, 2) == 6);
    matrix_print(mtrx4);
    printf("\n");

    Matrix *mtrx5 = matrix_create(3, 3);
    matrix_set_entry(mtrx5, 1, 1, -2);
    matrix_set_entry(mtrx5, 1, 2, 0);
    matrix_set_entry(mtrx5, 1, 3, 2);
    matrix_set_entry(mtrx5, 2, 1, 3);
    matrix_set_entry(mtrx5, 2, 2, 2);
    matrix_set_entry(mtrx5, 2, 3, 3);
    matrix_set_entry(mtrx5, 3, 1, 4);
    matrix_set_entry(mtrx5, 3, 2, 10);
    matrix_set_entry(mtrx5, 3, 3, 4);
    assert(matrix_get_entry(mtrx5, 1, 2) == 0);
    assert(matrix_get_entry(mtrx5, 2, 3) == 3);
    assert(matrix_get_entry(mtrx5, 3, 2) == 10);
    matrix_print(mtrx5);
    printf("\n");

    Matrix *mtrx6 = matrix_create(4, 4);
    matrix_set_entry(mtrx6, 1, 1, 2);
    matrix_set_entry(mtrx6, 1, 2, -2);
    matrix_set_entry(mtrx6, 1, 3, -4);
    matrix_set_entry(mtrx6, 1, 4, 9);
    matrix_set_entry(mtrx6, 2, 1, 5);
    matrix_set_entry(mtrx6, 2, 2, -3);
    matrix_set_entry(mtrx6, 2, 3, 2);
    matrix_set_entry(mtrx6, 2, 4, 1);
    matrix_set_entry(mtrx6, 3, 1, 6);
    matrix_set_entry(mtrx6, 3, 2, 10);
    matrix_set_entry(mtrx6, 3, 3, 1);
    matrix_set_entry(mtrx6, 3, 4, 1);
    matrix_set_entry(mtrx6, 4, 1, 7);
    matrix_set_entry(mtrx6, 4, 2, 11);
    matrix_set_entry(mtrx6, 4, 3, 0);
    matrix_set_entry(mtrx6, 4, 4, 9);
    assert(matrix_get_entry(mtrx6, 1, 4) == 9);
    assert(matrix_get_entry(mtrx6, 2, 2) == -3);
    assert(matrix_get_entry(mtrx6, 4, 2) == 11);
    matrix_print(mtrx6);
    printf("\n");

    Matrix *mtrx7 = matrix_create(3, 3);
    matrix_set_entry(mtrx7, 1, 1, 1);
    matrix_set_entry(mtrx7, 1, 2, 0);
    matrix_set_entry(mtrx7, 1, 3, 0);
    matrix_set_entry(mtrx7, 2, 1, 2);
    matrix_set_entry(mtrx7, 2, 2, 9);
    matrix_set_entry(mtrx7, 2, 3, 11);
    matrix_set_entry(mtrx7, 3, 1, 3);
    matrix_set_entry(mtrx7, 3, 2, 10);
    matrix_set_entry(mtrx7, 3, 3, 8);
    assert(matrix_get_entry(mtrx7, 1, 2) == 0);
    assert(matrix_get_entry(mtrx7, 2, 3) == 11);
    assert(matrix_get_entry(mtrx7, 3, 2) == 10);
    matrix_print(mtrx7);
    printf("\n");

    double *row1 = matrix_get_row(mtrx2, 1);
    double *row2 = matrix_get_row(mtrx6, 3);
    assert(dot_product(row1, row2, 4) == 33);

    double *row3 = matrix_get_row(mtrx3, 2);
    double *row4 = matrix_get_row(mtrx1, 2);
    assert(dot_product(row3, row4, 2) == 11);

    double *row5 = matrix_get_row(mtrx2, 3);
    double *col1 = matrix_get_col(mtrx6, 2);
    assert(dot_product(row5, col1, 4) == 52);
    free(col1);

    double *row6 = matrix_get_row(mtrx4, 1);
    double *col2 = matrix_get_col(mtrx7, 3);
    assert(dot_product(row6, col2, 3) == 39);
    free(col2);

    printf("==Multiplication==\n\n");

    Matrix *mult1 = matrix_multiply(mtrx1, mtrx4);
    matrix_print(mult1);
    printf("\n");
    matrix_destroy(mult1);
    
    Matrix *mult2 = matrix_multiply(mtrx2, mtrx3);
    matrix_print(mult2);
    printf("\n");
    matrix_destroy(mult2);

    Matrix *mult3 = matrix_multiply(mtrx4, mtrx5);
    matrix_print(mult3);
    printf("\n");
    matrix_destroy(mult3);

    Matrix *mult4 = matrix_multiply(mtrx5, mtrx2);
    matrix_print(mult4);
    printf("\n");
    matrix_destroy(mult4);

    printf("==Removing i-th Row and j-th Column==\n\n");

    Matrix *a1 = matrix_ij(mtrx5, 2, 2);
    matrix_print(a1);
    printf("\n");
    matrix_destroy(a1);

    Matrix *a2 = matrix_ij(mtrx2, 1, 1);
    matrix_print(a2);
    printf("\n");
    matrix_destroy(a2);

    Matrix *a3 = matrix_ij(mtrx2, 1, 4);
    matrix_print(a3);
    printf("\n");
    matrix_destroy(a3);

    Matrix *a6 = matrix_ij(mtrx2, 2, 3);
    matrix_print(a6);
    printf("\n");
    matrix_destroy(a6);

    Matrix *a4 = matrix_ij(mtrx6, 2, 2);
    matrix_print(a4);
    printf("\n");
    matrix_destroy(a4);

    Matrix *a5 = matrix_ij(mtrx6, 3, 3);
    matrix_print(a5);
    printf("\n");
    matrix_destroy(a5);

    printf("==Transpose==\n\n");

    Matrix *t1 = matrix_transpose(mtrx3);
    matrix_print(t1);
    printf("\n");
    matrix_destroy(t1);

    Matrix *t2 = matrix_transpose(mtrx4);
    matrix_print(t2);
    printf("\n");
    matrix_destroy(t2);

    Matrix *t3 = matrix_transpose(mtrx2);
    matrix_print(t3);
    printf("\n");
    matrix_destroy(t3);

    //This serves as a test of matrix_cofactor as well.
    printf("==Determinant==\n\n");

    printf("%g\n", matrix_determinant(mtrx7));
    printf("%g\n", matrix_determinant(mtrx1));
    printf("%g\n", matrix_determinant(mtrx5));
    printf("%g\n", matrix_determinant(mtrx6));

    printf("\n");

    printf("==Cofactor Matrices==\n\n");

    Matrix *c1 = matrix_cofactor(mtrx5);
    matrix_print(c1);
    printf("\n");
    matrix_destroy(c1);

    Matrix *c2 = matrix_cofactor(mtrx6);
    matrix_print(c2);
    printf("\n");
    matrix_destroy(c2);

    Matrix *c3 = matrix_cofactor(mtrx7);
    matrix_print(c3);
    printf("\n");
    matrix_destroy(c3);

    printf("==Adjugates==\n\n");

    Matrix *adj1 = matrix_adjugate(mtrx1);
    matrix_print(adj1);
    printf("\n");
    matrix_destroy(adj1);

    Matrix *adj2 = matrix_adjugate(mtrx5);
    matrix_print(adj2);
    printf("\n");
    matrix_destroy(adj2);

    Matrix *adj3 = matrix_adjugate(mtrx6);
    matrix_print(adj3);
    printf("\n");
    matrix_destroy(adj3);

    Matrix *adj4 = matrix_adjugate(mtrx7);
    matrix_print(adj4);
    printf("\n");
    matrix_destroy(adj4);

    printf("==Inverses==\n\n");

    //We check to see if the matrix multiplied by its inverse is the identity matrix.
    //Recall that there is unique inverse of every matrix, so if we do indeed get the identity matrix, we have correctly calculated the inverse.
    //It works, but doubles are not really nice to work with for this.

    Matrix *inv1 = matrix_inverse(mtrx1);
    Matrix *prod1 = matrix_multiply(mtrx1, inv1); 
    matrix_print(prod1);
    printf("\n");
    matrix_destroy(inv1);
    matrix_destroy(prod1);

    Matrix *inv2 = matrix_inverse(mtrx5);
    Matrix *prod2 = matrix_multiply(mtrx5, inv2); 
    matrix_print(prod2);
    printf("\n");
    matrix_destroy(inv2);
    matrix_destroy(prod2);

    Matrix *inv3 = matrix_inverse(mtrx6);
    Matrix *prod3 = matrix_multiply(mtrx6, inv3); 
    matrix_print(prod3);
    printf("\n");
    matrix_destroy(inv3);
    matrix_destroy(prod3);

    Matrix *inv4 = matrix_inverse(mtrx7);
    Matrix *prod4 = matrix_multiply(mtrx7, inv4); 
    matrix_print(prod4);
    printf("\n");
    matrix_destroy(inv4);
    matrix_destroy(prod4);

    printf("==Powers==\n\n");

    Matrix *p1 = matrix_power(mtrx1, 20);
    matrix_print(p1);
    printf("\n");
    matrix_destroy(p1);

    Matrix *p2 = matrix_power(mtrx5, 15);
    matrix_print(p2);
    printf("\n");
    matrix_destroy(p2);

    Matrix *p3 = matrix_power(mtrx6, 20);
    matrix_print(p3);
    printf("\n");
    matrix_destroy(p3);

    Matrix *p4 = matrix_power(mtrx7, 5);
    matrix_print(p4);
    printf("\n");
    matrix_destroy(p4);

    printf("==Addition==\n\n");

    mtrx5 = matrix_add(mtrx5, mtrx7);
    matrix_print(mtrx5);
    printf("\n");

    printf("==Scalar Multiplication==\n\n");

    mtrx3 = matrix_scalar_multiply(mtrx3, 3);
    matrix_print(mtrx3);
    printf("\n");

    matrix_destroy(mtrx1);
    matrix_destroy(mtrx2);
    matrix_destroy(mtrx3);
    matrix_destroy(mtrx4);
    matrix_destroy(mtrx5);
    matrix_destroy(mtrx6);
    matrix_destroy(mtrx7);
}