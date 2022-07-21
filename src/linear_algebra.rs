#[derive(Clone)]
pub struct Tuple4 {
    data: [f64; 4],
}

pub fn vector(x: f64, y: f64, z: f64) -> Tuple4 {
    Tuple4 {
        data: [x, y, z, 0.0],
    }
}

pub fn point(x: f64, y: f64, z: f64) -> Tuple4 {
    Tuple4 {
        data: [x, y, z, 1.0],
    }
}

pub fn f64_approx_eq(a: f64, b: f64) -> bool {
    f64::abs(a - b) < 0.00000001
}

pub fn normalized(t: &Tuple4) -> Tuple4 {
    let mut new_t = t.clone();
    new_t.normalize();
    new_t
}

pub fn dot(a: &Tuple4, b: &Tuple4) -> f64 {
    std::iter::zip(a.data, b.data).map(|(c1, c2)| c1 * c2).sum()
}

pub fn cross(a: &Tuple4, b: &Tuple4) -> Tuple4 {
    vector(
        a.y() * b.z() - a.z() * b.y(),
        a.z() * b.x() - a.x() * b.z(),
        a.x() * b.y() - a.y() * b.x()
    )
}

impl Tuple4 {
    pub fn new(x: f64, y: f64, z: f64, w: f64) -> Tuple4 {
        Tuple4 { data: [x, y, z, w] }
    }

    pub fn x(&self) -> f64 {
        self.data[0]
    }
    pub fn y(&self) -> f64 {
        self.data[1]
    }
    pub fn z(&self) -> f64 {
        self.data[2]
    }
    pub fn w(&self) -> f64 {
        self.data[3]
    }

    pub fn magnitude(&self) -> f64 {
        f64::sqrt(self.data.iter().map(|t| t * t).sum())
    }

    pub fn normalize(&mut self) {
        let m = self.magnitude();
        self.data.iter_mut().for_each(|coord| *coord /= m);
    }
}

impl std::cmp::PartialEq for Tuple4 {
    fn eq(&self, other: &Self) -> bool {
        std::iter::zip(self.data, other.data).all(|(l, r)| f64::abs(l - r) < 0.00000001)
    }
}

impl std::cmp::Eq for Tuple4 {}

impl std::ops::Add<&Tuple4> for &Tuple4 {
    type Output = Tuple4;

    fn add(self, other: &Tuple4) -> Self::Output {
        Tuple4::new(
            self.x() + other.x(),
            self.y() + other.y(),
            self.z() + other.z(),
            self.w() + other.w(),
        )
    }
}

impl std::ops::Sub<&Tuple4> for &Tuple4 {
    type Output = Tuple4;

    fn sub(self, other: &Tuple4) -> Self::Output {
        Tuple4::new(
            self.x() - other.x(),
            self.y() - other.y(),
            self.z() - other.z(),
            self.w() - other.w(),
        )
    }
}

impl std::ops::Neg for &Tuple4 {
    type Output = Tuple4;

    fn neg(self) -> Self::Output {
        Tuple4::new(-self.x(), -self.y(), -self.z(), -self.w())
    }
}

impl std::ops::Mul<f64> for &Tuple4 {
    type Output = Tuple4;

    fn mul(self, scalar: f64) -> Self::Output {
        let mut new_data: [f64; 4] = self.data.clone();
        new_data.iter_mut().for_each(|coord| (*coord) *= scalar);
        Tuple4 { data: new_data }
    }
}

impl std::ops::Div<f64> for &Tuple4 {
    type Output = Tuple4;

    fn div(self, scalar: f64) -> Self::Output {
        let mut new_data: [f64; 4] = self.data.clone();
        new_data.iter_mut().for_each(|coord| (*coord) /= scalar);
        Tuple4 { data: new_data }
    }
}


impl std::fmt::Debug for Tuple4 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(format!("({}, {}, {}, {})", self.x(), self.y(), self.z(), self.w()).as_str())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adding_two_tuples() {
        let a1 = Tuple4::new(3.0, -2.0, 5.0, 1.0);
        let a2 = Tuple4::new(-2.0, 3.0, 1.0, 0.0);

        assert_eq!(&a1 + &a2, Tuple4::new(1.0, 1.0, 6.0, 1.0));
    }

    #[test]
    fn subtracting_two_points_gives_vector() {
        let p1 = point(3.0, 2.0, 1.0);
        let p2 = point(5.0, 6.0, 7.0);

        assert_eq!(&p1 - &p2, vector(-2.0, -4.0, -6.0));
    }

    #[test]
    fn subtracting_vector_from_point() {
        let p = point(3.0, 2.0, 1.0);
        let v = vector(5.0, 6.0, 7.0);

        assert_eq!(&p - &v, point(-2.0, -4.0, -6.0))
    }

    #[test]
    fn subtracting_two_vectors() {
        let p = vector(3.0, 2.0, 1.0);
        let v = vector(5.0, 6.0, 7.0);

        assert_eq!(&p - &v, vector(-2.0, -4.0, -6.0))
    }

    #[test]
    fn subtracting_vector_from_zero() {
        let zero = vector(0.0, 0.0, 0.0);
        let v = vector(1.0, -2.0, 3.0);

        assert_eq!(&zero - &v, vector(-1.0, 2.0, -3.0));
    }

    #[test]
    fn negating_tuple() {
        let a = Tuple4::new(1.0, -2.0, 3.0, -4.0);

        assert_eq!(-&a, Tuple4::new(-1.0, 2.0, -3.0, 4.0));
    }

    #[test]
    fn multiply_tuple_by_scalar() {
        let a = Tuple4::new(1.0, -2.0, 3.0, -4.0);

        assert_eq!(&a * 3.5, Tuple4::new(3.5, -7.0, 10.5, -14.0));
    }

    #[test]
    fn multiply_tuple_by_fraction() {
        let a = Tuple4::new(1.0, -2.0, 3.0, -4.0);

        assert_eq!(&a * 0.5, Tuple4::new(0.5, -1.0, 1.5, -2.0));
    }

    #[test]
    fn dividing_tuple_by_scalar() {
        let a = Tuple4::new(1.0, -2.0, 3.0, -4.0);

        assert_eq!(&a / 2.0, Tuple4::new(0.5, -1.0, 1.5, -2.0));
    }

    #[test]
    fn magnitude_1_0_0_is_1() {
        let v = vector(1.0, 0.0, 0.0);
        let expected_magnitude = 1.0;

        assert!(f64_approx_eq(v.magnitude(), expected_magnitude));
    }

    #[test]
    fn magnitude_0_1_0_is_1() {
        let v = vector(0.0, 1.0, 0.0);
        let expected_magnitude = 1.0;

        assert!(f64_approx_eq(v.magnitude(), expected_magnitude));
    }

    #[test]
    fn magnitude_0_0_1_is_1() {
        let v = vector(0.0, 0.0, 1.0);
        let expected_magnitude = 1.0;

        assert!(f64_approx_eq(v.magnitude(), expected_magnitude));
    }

    #[test]
    fn magnitude_1_2_3_is_sqrt14() {
        let v = vector(1.0, 2.0, 3.0);
        let expected_magnitude = f64::sqrt(14.0);

        assert!(f64_approx_eq(v.magnitude(), expected_magnitude));
    }

    #[test]
    fn magnitude_neg1_neg2_neg3_is_sqrt14() {
        let v = vector(-1.0, -2.0, -3.0);
        let expected_magnitude = f64::sqrt(14.0);

        assert!(f64_approx_eq(v.magnitude(), expected_magnitude));
    }

    #[test]
    fn normalizing_4_0_0_gives_1_0_0() {
        let v = vector(4.0, 0.0, 0.0);

        assert_eq!(normalized(&v), vector(1.0, 0.0, 0.0))
    }

    #[test]
    fn normalizing_1_2_3_gives_stuff() {
        let v = vector(1.0, 2.0, 3.0);

        assert_eq!(normalized(&v), vector(0.267261241, 0.534522483, 0.801783725));
    }

    #[test]
    fn normalized_vector_has_magnitude_1() {
        let v = vector(1.0, 2.0, 3.0);
        let normalized_v = normalized(&v);
        let expected_magnitude = 1.0;

        assert!(f64_approx_eq(normalized_v.magnitude(), expected_magnitude));
    }

    #[test]
    fn dot_product_of_tuples() {
        let a = vector(1.0, 2.0, 3.0);
        let b = vector(2.0, 3.0, 4.0);

        assert!(f64_approx_eq(dot(&a, &b), 20.0))
    }

    #[test]
    fn cross_product_of_vectors() {
        let a = vector(1.0, 2.0, 3.0);
        let b = vector(2.0, 3.0, 4.0);

        assert_eq!(cross(&a, &b), vector(-1.0, 2.0, -1.0));
        assert_eq!(cross(&b, &a), vector(1.0, -2.0, 1.0));
    }
}
