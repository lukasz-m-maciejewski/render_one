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

pub trait Sqrt {
    fn sqrt(&self) -> Self;
}

impl Sqrt for f64 {
    fn sqrt(&self) -> f64 {
        f64::sqrt(*self)
    }
}

impl Sqrt for f32 {
    fn sqrt(&self) -> f32 {
        f32::sqrt(*self)
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
    + std::fmt::Debug
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
            + std::ops::DivAssign
            + std::fmt::Debug,
    > Field for T
{
}

#[derive(Clone)]
pub struct Matrix<const N: usize, const M: usize, T: Field> {
    data: Vec<T>,
}

impl<const N: usize, const M: usize, T: Field> std::fmt::Debug for Matrix<N, M, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("[\n")?;
        for i in 0..N {
            f.write_str("[")?;
            f.write_fmt(format_args!("{:?}", self[(i, 0)]))?;
            for j in 1..M {
                f.write_fmt(format_args!(",{:?}", self[(i, j)]))?;
            }
            f.write_str("],\n")?;
        }
        f.write_str("]")
    }
}

impl<const N: usize, const M: usize, T: Field> std::fmt::Display for Matrix<N, M, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("[\n")?;
        for i in 0..N {
            f.write_str("[")?;
            f.write_fmt(format_args!("{:?}", self[(i, 0)]))?;
            for j in 1..M {
                f.write_fmt(format_args!(",{:?}", self[(i, j)]))?;
            }
            f.write_str("],\n")?;
        }
        f.write_str("]")
    }
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

pub fn dot<const N: usize, T: Field>(a: &Matrix<N, 1, T>, b: &Matrix<N, 1, T>) -> T {
    let mut prod = T::zero();
    for i in 0..N {
        prod += a[i] * b[i];
    }
    prod
}

pub fn norm_squared<const N: usize, const M: usize, T: Field>(m: &Matrix<N, M, T>) -> T {
    let mut norm = T::zero();
    for i in 0..N {
        for j in 0..M {
            norm += m[(i, j)] * m[(i, j)];
        }
    }
    norm
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

impl<const N: usize, T: Field> std::ops::Index<usize> for Matrix<N, 1, T> {
    type Output = T;
    fn index(&self, index: usize) -> &Self::Output {
        assert!(index < N, "index out of bounds: {}", index);
        &self.data[index]
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

impl<const N: usize, T: Field> std::ops::IndexMut<usize> for Matrix<N, 1, T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        assert!(index < N, "index out of bounds: {}", index);
        &mut self.data[index]
    }
}

impl<const N: usize, const M: usize, T: Field> std::cmp::PartialEq for Matrix<N, M, T> {
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
{
    fn add_assign(&mut self, rhs: &Matrix<N, M, T>) {
        std::iter::zip(self.data.iter_mut(), rhs.data.iter()).for_each(|(a, b)| *a += *b)
    }
}

impl<const N: usize, const M: usize, T: Field> std::ops::Add<&Matrix<N, M, T>>
    for &Matrix<N, M, T>
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

impl<const N: usize, const M: usize, const P: usize, T: Field> std::ops::Mul<&Matrix<M, P, T>>
    for &Matrix<N, M, T>
{
    type Output = Matrix<N, P, T>;

    fn mul(self, rhs: &Matrix<M, P, T>) -> Self::Output {
        let mut out = Matrix::<N, P, T>::default();
        for i in 0..N {
            for j in 0..P {
                for k in 0..M {
                    out[(i, j)] = out[(i, j)] + self[(i, k)] * rhs[(k, j)];
                }
            }
        }
        out
    }
}

pub enum QRResult<const N: usize, T: Field> {
    Singular,
    Decomposition(Matrix<N, N, T>, Matrix<N, N, T>),
}

pub fn qr_decomposition<const N: usize, T: Field + Sqrt>(m: &Matrix<N, N, T>) -> QRResult<N, T> {
    let mut q = m.clone();
    let mut r = Matrix::<N, N, T>::default();

    for k in 0..N {
        println!("k:{}", k);
        for i in 0..k {
            println!("  i:{}", i);
            r[(i, k)] = column_product(&q, i, k);
            column_sub_assign_with_factor(&mut q, k, i, r[(i, k)]);
        }

        r[(k, k)] = column_norm(&q, k);
        if (r[(k, k)]).approx_eq(&T::zero()) {
            return QRResult::Singular;
        }
        column_div_assign(&mut q, k, r[(k, k)]);
        println!("q:{}", q);
        println!("r:{}", r);
    }

    QRResult::Decomposition(-&q, -&r)
    //QRResult::Decomposition(q, r)
}

fn column_product<const N: usize, T: Field>(m: &Matrix<N, N, T>, idx1: usize, idx2: usize) -> T {
    let mut prod = T::zero();
    for i in 0..N {
        prod += m[(i, idx1)] * m[(i, idx2)];
    }
    prod
}

fn column_norm<const N: usize, T: Field + Sqrt>(m: &Matrix<N, N, T>, idx: usize) -> T {
    let mut norm = T::zero();
    for i in 0..N {
        norm += m[(i, idx)] * m[(i, idx)];
    }

    norm.sqrt()
}

fn column_sub_assign_with_factor<const N: usize, T: Field>(
    m: &mut Matrix<N, N, T>,
    modified_col: usize,
    from: usize,
    factor: T,
) {
    for i in 0..N {
        let from_coeff = m[(i, from)];
        m[(i, modified_col)] -= factor * from_coeff;
    }
}

fn column_div_assign<const N: usize, T: Field>(m: &mut Matrix<N, N, T>, idx: usize, divisor: T) {
    for i in 0..N {
        m[(i, idx)] /= divisor;
    }
}

pub fn lup_decomposition<const N: usize, T: Field>(
    a: &Matrix<N, N, T>,
) -> (Matrix<N, N, T>, Matrix<N, N, T>, [usize; N + 1]) {
    let mut permutations: [usize; N + 1] = [0; N + 1];
    for i in 0..(N + 1) {
        permutations[i] = i;
    }

    for i in 0..N {}

    (a.clone(), a.clone(), permutations)
}

fn swap_columns<const N: usize, T: Field>(a: &mut Matrix<N, N, T>, idx1: usize, idx2: usize) {
    for i in 0..N {
        let tmp = a[(i, idx1)];
        a[(i, idx1)] = a[(i, idx2)];
        a[(i, idx2)] = tmp;
    }
}

pub fn determinant<const N: usize, T: Field + Sqrt>(a: &Matrix<N, N, T>) -> T {
    if let QRResult::Decomposition(_, r) = qr_decomposition(a) {
        //let r = -&r;
        let mut det = T::one();
        for i in 0..N {
            det *= r[(i, i)];
        }
        return det;
    } else {
        return T::zero();
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
        let m = Matrix::<4, 4, _>::from_nested([
            [0, 1, 2, 4],
            [1, 2, 4, 8],
            [2, 4, 8, 16],
            [4, 8, 16, 32],
        ]);
        let id = identity::<4, i32>();

        assert_eq!(&m * &id, m);
        assert_eq!(&id * &m, m);
    }

    #[test]
    fn matrix_transpose_identity() {
        let id = identity::<4, i32>();
        let id_transposed = id.transposed();

        assert_eq!(id, id_transposed);
    }

    #[test]
    fn matrix_vector_single_index() {
        let v = Matrix::<3, 1, _>::new([1, 2, 3]);

        assert_eq!(v[0], 1);
        assert_eq!(v[1], 2);
        assert_eq!(v[2], 3);
        assert_eq!(v[(0, 0)], 1);
        assert_eq!(v[(1, 0)], 2);
        assert_eq!(v[(2, 0)], 3);
    }

    #[test]
    fn matrix_qr_decomposition() {
        let a = Matrix::<3, 3, f64>::from_nested([
            [12.0, -51.0, 4.0],
            [6.0, 167.0, -68.0],
            [-4.0, 24.0, -41.0],
        ]);

        let expected_q = Matrix::<3, 3, f64>::from_nested([
            [-6.0 / 7.0, 69.0 / 175.0, 58.0 / 175.0],
            [-3.0 / 7.0, -158.0 / 175.0, -6.0 / 175.0],
            [2.0 / 7.0, -6.0 / 35.0, 33.0 / 35.0],
        ]);

        let expected_r = Matrix::<3, 3, f64>::from_nested([
            [-14.0, -21.0, 14.0],
            [0.0, -175.0, 70.0],
            [0.0, 0.0, -35.0],
        ]);

        if let QRResult::Decomposition(q, r) = qr_decomposition(&a) {
            assert_eq!(q, expected_q);
            assert_eq!(r, expected_r);
        } else {
            assert!(false);
        }
    }

    #[test]
    fn matrix_qr_decomposition_singular_matrix() {
        let a = Matrix::<3, 3, f64>::from_nested([
            [12.0, -51.0, 4.0],
            [0.0, 0.0, 0.0],
            [6.0, 167.0, -68.0],
        ]);

        if let QRResult::Singular = qr_decomposition(&a) {
            assert!(true);
        } else {
            assert!(false);
        }
    }

    #[test]
    fn matrix_determinant_identity_is_one() {
        let id = identity::<4, f64>();
        let det = determinant(&id);

        assert_eq!(det, 1.0);
    }

    #[test]
    fn matrix_determinant_singular_is_zero() {
        let a = Matrix::<3, 3, f64>::from_nested([
            [12.0, -51.0, 4.0],
            [0.0, 0.0, 0.0],
            [6.0, 167.0, -68.0],
        ]);

        let det = determinant(&a);
        assert!(det.approx_eq(&0.0));
    }

    #[test]
    fn matrix_determinant_2x2_is_correct() {
        let a = Matrix::<2, 2, f64>::new([3., 8., 4., 6.]);
        let det: f64 = determinant(&a);

        assert!(det.approx_eq(&(-14.)), "{}", det);
    }

    #[test]
    fn matrix_determinant_2x2_another_is_correct() {
        let a = Matrix::<2, 2, f64>::new([1., 2., 3., 4.]);
        let det: f64 = determinant(&a);

        assert!((-2.0).approx_eq(&det), "det:{}", det);
    }

    #[test]
    fn matrix_determinant_3x3_is_correct() {
        let a = Matrix::<3, 3, f64>::new([6.0, 1.0, 1.0, 4.0, -2.0, 5.0, 2.0, 8.0, 7.0]);
        let det: f64 = determinant(&a);

        assert!((-306.0).approx_eq(&det), "det:{}", det);
    }

    #[test]
    fn matrix_qr_decomposition_2() {
        let a =
            Matrix::<3, 3, f64>::from_nested([[6.0, 1.0, 1.0], [4.0, -2.0, 5.0], [2.0, 8.0, 7.0]]);

        let expected_q = Matrix::<3, 3, f64>::from_nested([
            [-0.80178373, -0.06178021, -0.59441237],
            [-0.53452248, -0.37068124, 0.75952691],
            [-0.26726124, 0.92670309, 0.26418327],
        ]);

        let expected_r = Matrix::<3, 3, f64>::from_nested([
            [-7.48331477, -1.87082869, -5.34522484],
            [0., 8.09320703, 4.57173527],
            [0., 0., 5.05250513],
        ]);

        if let QRResult::Decomposition(q, r) = qr_decomposition(&a) {
            assert_eq!(q, expected_q, "Q:{}", q);
            assert_eq!(r, expected_r, "R:{}", r);
        } else {
            assert!(false);
        }
    }

    #[test]
    fn matrix_qr_decomposition_3() {
        let a = Matrix::<2, 2, f64>::from_nested([[1., 2.], [3., 4.]]);

        let expected_q =
            Matrix::<2, 2, f64>::from_nested([[-0.31622777, -0.9486833], [-0.9486833, 0.31622777]]);

        let expected_r =
            Matrix::<2, 2, f64>::from_nested([[-3.16227766, -4.42718872], [0., -0.63245553]]);
        if let QRResult::Decomposition(q, r) = qr_decomposition(&a) {
            assert!(q == expected_q, "Q:{}", q);
            assert!(r == expected_r, "R:{}", r);
        } else {
            assert!(false);
        }
    }
}
