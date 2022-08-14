use crate::float_compare::{f32_approx_eq, f64_approx_eq};

pub trait ApproxEq {
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

pub trait Zero {
    fn zero() -> Self;
}

impl Zero for f64 {
    fn zero() -> f64 {
        0.0
    }
}

impl Zero for f32 {
    fn zero() -> f32 {
        0.0
    }
}

impl Zero for i32 {
    fn zero() -> i32 {
        0
    }
}

pub trait One {
    fn one() -> Self;
}

impl One for f64 {
    fn one() -> f64 {
        1.0
    }
}

impl One for f32 {
    fn one() -> f32 {
        1.0
    }
}

impl One for i32 {
    fn one() -> i32 {
        1
    }
}

pub trait Field:
    Copy
    + ApproxEq
    + std::default::Default
    + Zero
    + std::ops::Add<Output = Self>
    + std::ops::AddAssign
    + std::ops::Neg<Output = Self>
    + std::ops::Sub<Output = Self>
    + std::ops::SubAssign
    + One
    + std::ops::Mul<Output = Self>
    + std::ops::MulAssign
    + std::ops::Div<Output = Self>
    + std::ops::DivAssign
{
}
impl<
        T: Copy
            + ApproxEq
            + std::default::Default
            + Zero
            + std::ops::Add<Output = T>
            + std::ops::AddAssign
            + std::ops::Neg<Output = T>
            + std::ops::Sub<Output = T>
            + std::ops::SubAssign
            + One
            + std::ops::Mul<Output = T>
            + std::ops::MulAssign
            + std::ops::Div<Output = Self>
            + std::ops::DivAssign,
    > Field for T
{
}

#[derive(Clone, Debug)]
pub struct Matrix<const N: usize, const M: usize, T: Field> {
    data: Vec<T>,
}

impl<const N: usize, const M: usize, T: Field> std::default::Default for Matrix<N, M, T> {
    fn default() -> Self {
        let mut data: Vec<T> = vec![];

        for _ in 0..N * M {
            data.push(T::default());
        }

        Matrix::<N, M, T> { data }
    }
}

impl<const N: usize, const M: usize, T: Field> Matrix<N, M, T> {
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

    pub fn transposed(&self) -> Matrix<M, N, T> {
        let mut m = Matrix::<M, N, T>::default();
        for i in 0..M {
            for j in 0..N {
                m[(i, j)] = self[(j, i)];
            }
        }

        m
    }

}

pub fn identity<const N: usize, T: Field>() -> Matrix<N, N, T> {
    let mut m = Matrix::<N, N, T>::default();
    for i in 0..N {
        m[(i, i)] = T::one();
    }

    m
}

impl<const N: usize, const M: usize, T: Field> FromIterator<T> for Matrix<N, M, T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let maybe_data = Vec::from_iter(iter);

        assert!(maybe_data.len() == (N * M));

        Matrix::<N, M, T> { data: maybe_data }
    }
}

impl<const N: usize, const M: usize, T: Field> std::ops::Index<(usize, usize)> for Matrix<N, M, T> {
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

impl<const N: usize, const M: usize, T: Field> std::ops::IndexMut<(usize, usize)>
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
    T: Field,
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

impl<const N: usize, const M: usize, T: Field> std::ops::AddAssign<&Matrix<N, M, T>>
    for Matrix<N, M, T>
where
    T: Field,
{
    fn add_assign(&mut self, rhs: &Matrix<N, M, T>) {
        std::iter::zip(self.data.iter_mut(), rhs.data.iter()).for_each(|(a, b)| *a += *b)
    }
}

impl<const N: usize, const M: usize, T: Field> std::ops::Add<&Matrix<N, M, T>> for &Matrix<N, M, T>
where
    T: Field,
{
    type Output = Matrix<N, M, T>;
    fn add(self, rhs: &Matrix<N, M, T>) -> Self::Output {
        let mut m = self.clone();
        m += rhs;
        m
    }
}

impl<const N: usize, const M: usize, T: Field> std::ops::SubAssign<&Matrix<N, M, T>>
    for Matrix<N, M, T>
where
    T: Field,
{
    fn sub_assign(&mut self, rhs: &Matrix<N, M, T>) {
        std::iter::zip(self.data.iter_mut(), rhs.data.iter()).for_each(|(a, b)| *a -= *b)
    }
}

impl<const N: usize, const M: usize, T: Field> std::ops::Sub<&Matrix<N, M, T>>
    for &Matrix<N, M, T>
{
    type Output = Matrix<N, M, T>;
    fn sub(self, rhs: &Matrix<N, M, T>) -> Self::Output {
        let mut m = self.clone();
        m -= rhs;
        m
    }
}

impl<const N: usize, const M: usize, T: Field> std::ops::Neg for &Matrix<N, M, T> {
    type Output = Matrix<N, M, T>;
    fn neg(self) -> Self::Output {
        let mut m = self.clone();
        m.data.iter_mut().for_each(|a| *a = a.neg());
        m
    }
}

impl<const N: usize, const M: usize, T: Field> std::ops::MulAssign<T> for Matrix<N, M, T> {
    fn mul_assign(&mut self, rhs: T) {
        self.data.iter_mut().for_each(|a| *a *= rhs);
    }
}

impl<const N: usize, const M: usize, T: Field> std::ops::Mul<T> for &Matrix<N, M, T> {
    type Output = Matrix<N, M, T>;
    fn mul(self, rhs: T) -> Self::Output {
        let mut out = self.clone();
        out.data.iter_mut().for_each(|a| *a *= rhs);
        out
    }
}

impl<const N: usize, const M: usize, T: Field> std::ops::DivAssign<T> for Matrix<N, M, T> {
    fn div_assign(&mut self, rhs: T) {
        self.data.iter_mut().for_each(|a| *a /= rhs);
    }
}

impl<const N: usize, const M: usize, T: Field> std::ops::Div<T> for &Matrix<N, M, T> {
    type Output = Matrix<N, M, T>;
    fn div(self, rhs: T) -> Self::Output {
        let mut out = self.clone();
        out.data.iter_mut().for_each(|a| *a /= rhs);
        out
    }
}

fn mul<const N: usize, const M: usize, const P: usize, T>(
    lhs: &Matrix<N, M, T>,
    rhs: &Matrix<M, P, T>,
) -> Matrix<N, P, T>
where
    T: Field,
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

impl<const N: usize, const M: usize, const P: usize, T: Field> std::ops::Mul<&Matrix<M, P, T>>
    for &Matrix<N, M, T>
{
    type Output = Matrix<N, P, T>;

    fn mul(self, rhs: &Matrix<M, P, T>) -> Self::Output {
        mul(self, rhs)
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
    fn matrix_transpose() {
        let m = Matrix::<2, 4, _>::from_nested([[1, 2, 3, 4], [5, 6, 7, 8]]);
        let m_transposed = Matrix::<4, 2, _>::from_nested([[1, 5], [2, 6], [3, 7], [4, 8]]);

        assert_eq!(m.transposed(), m_transposed);
    }

    #[test]
    fn matrix_add() {
        let m1 = Matrix::<2, 2, _>::from_nested([[1, 2], [3, 4]]);
        let m2 = Matrix::<2, 2, _>::from_nested([[1, 1], [1, 1]]);

        let expected = Matrix::<2, 2, _>::from_nested([[2, 3], [4, 5]]);

        assert_eq!(&m1 + &m2, expected);
    }

    #[test]
    fn matrix_add_assign() {
        let mut m1 = Matrix::<2, 2, _>::from_nested([[1, 2], [3, 4]]);
        let m2 = Matrix::<2, 2, _>::from_nested([[1, 1], [1, 1]]);

        let expected = Matrix::<2, 2, _>::from_nested([[2, 3], [4, 5]]);

        m1 += &m2;

        assert_eq!(m1, expected);
    }

    #[test]
    fn matrix_multiplication() {
        let m1 = Matrix::<4, 3, _>::new([1, 0, 1, 2, 1, 1, 0, 1, 1, 1, 1, 2]);
        let m2 = Matrix::<3, 3, _>::new([1, 2, 1, 2, 3, 1, 4, 2, 2]);

        let prod = &m1 * &m2;
        let expected_prod = Matrix::<4, 3, _>::new([5, 4, 3, 8, 9, 5, 6, 5, 3, 11, 9, 6]);

        assert_eq!(prod, expected_prod);
    }

    #[test]
    fn matrix_multiplicative_identity() {
        let m = Matrix::<4, 4, _>::from_nested([[0, 1, 2, 4 ], [1, 2, 4, 8], [2, 4, 8, 16], [4, 8, 16, 32]]);
        let id = identity::<4, i32>();

        assert_eq!(&m * &id, m);
        assert_eq!(&id * &m, m);
    }
}
