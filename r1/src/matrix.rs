use std::ops::{Add, AddAssign, Mul, MulAssign};

use crate::float_compare::{f32_approx_eq, f64_approx_eq};

trait ApproxEq {
    fn approx_eq(&self, other: &Self) -> bool;
}

impl ApproxEq for f64 {
    fn approx_eq(&self, other: &f64) -> bool {
        f64_approx_eq(*self, *other)
    }
}

impl ApproxEq for f32 {
    fn approx_eq(&self, other: &f32) -> bool {
        f32_approx_eq(*self, *other)
    }
}

impl ApproxEq for i32 {
    fn approx_eq(&self, other: &i32) -> bool {
        *self == *other
    }
}

trait ArithmeticType: Copy + std::default::Default + Add + Mul + AddAssign + MulAssign {}
impl<
        T: Copy + std::default::Default + Add<Output = T> + Mul<Output = T> + AddAssign + MulAssign,
    > ArithmeticType for T
{
}

#[derive(Clone, Debug)]
pub struct Matrix<const N: usize, const M: usize, T: ArithmeticType> {
    data: Vec<T>,
}

impl<const N: usize, const M: usize, T: ArithmeticType> std::default::Default for Matrix<N, M, T> {
    fn default() -> Self {
        let mut data: Vec<T> = vec![];

        for _ in 0..N * M {
            data.push(T::default());
        }

        Matrix::<N, M, T> { data }
    }
}

impl<const N: usize, const M: usize, T: ArithmeticType> Matrix<N, M, T> {
    pub fn new(coeffs: [T; N * M]) -> Matrix<N, M, T> {
        Matrix::<N, M, T> {
            data: coeffs.to_vec(),
        }
    }

    pub fn from_nested(nested_coeffs: [[T; M]; N]) -> Matrix<N, M, T> {
        let mut data: Vec<T> = vec![];

        for coeffs in nested_coeffs {
            for c in coeffs {
                data.push(c);
            }
        }

        Matrix::<N, M, T> { data }
    }
}

impl<const N: usize, const M: usize, T: ArithmeticType> FromIterator<T> for Matrix<N, M, T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let maybe_data = Vec::from_iter(iter);

        assert!(maybe_data.len() == (N * M));

        Matrix::<N, M, T> { data: maybe_data }
    }
}

impl<const N: usize, const M: usize, T: ArithmeticType> std::ops::Index<(usize, usize)>
    for Matrix<N, M, T>
{
    type Output = T;
    fn index(&self, index: (usize, usize)) -> &Self::Output {
        assert!(
            index.0 < N && index.1 < M,
            "index out of bounds: ({}, {})",
            index.0,
            index.1
        );
        &self.data[index.0 * M + index.1]
    }
}

impl<const N: usize, const M: usize, T: ArithmeticType> std::ops::IndexMut<(usize, usize)>
    for Matrix<N, M, T>
{
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        assert!(
            index.0 < N && index.1 < M,
            "index out of bounds: ({}, {})",
            index.0,
            index.1
        );
        &mut self.data[index.0 * M + index.1]
    }
}

impl<const N: usize, const M: usize, T: Copy> std::cmp::PartialEq for Matrix<N, M, T>
where
    T: ArithmeticType + ApproxEq,
{
    fn eq(&self, other: &Self) -> bool {
        for i in 0..({ N * M }) {
            if !&(self.data[i]).approx_eq(&other.data[i]) {
                return false;
            }
        }

        true
    }
}

impl<const N: usize, const M: usize, const P: usize, T: ArithmeticType>
    std::ops::AddAssign<Matrix<M, P, T>> for Matrix<N, M, T>
where
    T: Copy + std::default::Default + std::ops::Add<Output = T>,
{
    fn add_assign(&mut self, rhs: Matrix<M, P, T>) {
        todo!()
    }
}

fn mul<const N: usize, const M: usize, const P: usize, T>(
    lhs: &Matrix<N, M, T>,
    rhs: &Matrix<M, P, T>,
) -> Matrix<N, P, T>
where
    T: ArithmeticType,
{
    let mut out = Matrix::<N, P, T>::default();
    for i in 0..N {
        for j in 0..P {
            for k in 0..M {
                out[(i, j)] = out[(i, j)] + lhs[(i, k)] * rhs[(k, j)];
            }
        }
    }

    out
}

impl<const N: usize, const M: usize, const P: usize, T: Copy> std::ops::Mul<Matrix<M, P, T>>
    for Matrix<N, M, T>
where
    T: ArithmeticType,
{
    type Output = Matrix<N, P, T>;

    fn mul(self, rhs: Matrix<M, P, T>) -> Self::Output {
        mul(&self, &rhs)
    }
}

pub type Matrix2 = Matrix<2, 2, f64>;
pub type Matrix3 = Matrix<3, 3, f64>;
pub type Matrix4 = Matrix<4, 4, f64>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matrix_create() {
        let m = Matrix::<2, 2, f64>::new([1.0, 2.0, 3.0, 4.0]);

        assert_eq!(m[(0, 0)], 1.0);
    }

    #[test]
    fn matrix2_create() {
        let m = Matrix2::new([1.1, 1.2, 1.3, 1.4]);

        assert_eq!(m[(0, 1)], 1.2);
        assert_eq!(m[(1, 0)], 1.3);
    }

    #[test]
    fn matrix_equality_compare() {
        let m1 = Matrix::<2, 2, f64>::new([1.0, 2.0, 3.0, 4.0]);
        let m2 = Matrix::<2, 2, f64>::new([1.0, 2.0, 3.0, 4.000000001]);
        let m3 = Matrix::<2, 2, f64>::new([1.0, 2.0, 3.0, 4.00001]);

        assert_eq!(m1, m2);
        assert_ne!(m1, m3);
    }

    #[test]
    fn matrix_multiplication() {
        let m1 = Matrix::<4, 3, _>::new([1, 0, 1, 2, 1, 1, 0, 1, 1, 1, 1, 2]);
        let m2 = Matrix::<3, 3, _>::new([1, 2, 1, 2, 3, 1, 4, 2, 2]);

        let prod = m1 * m2;
        let expected_prod = Matrix::<4, 3, _>::new([5, 4, 3, 8, 9, 5, 6, 5, 3, 11, 9, 6]);

        assert_eq!(prod, expected_prod);
    }
}
