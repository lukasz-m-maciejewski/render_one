use crate::matrix::*;

pub type Tuple4 = Matrix<4, 1, f64>;

pub type Vector = Tuple4;
pub type Point = Tuple4;

pub fn vector(x: f64, y: f64, z: f64) -> Tuple4 {
    Tuple4::new([x, y, z, 0.0])
}

pub fn point(x: f64, y: f64, z: f64) -> Tuple4 {
    Tuple4::new([x, y, z, 1.0])
}

pub fn cross(a: &Tuple4, b: &Tuple4) -> Tuple4 {
    vector(
        a.y() * b.z() - a.z() * b.y(),
        a.z() * b.x() - a.x() * b.z(),
        a.x() * b.y() - a.y() * b.x(),
    )
}

impl Tuple4 {
    pub fn x(&self) -> f64 {
        self[(0, 0)]
    }
    pub fn y(&self) -> f64 {
        self[(1, 0)]
    }
    pub fn z(&self) -> f64 {
        self[(2, 0)]
    }
    pub fn w(&self) -> f64 {
        self[(3, 0)]
    }

    pub fn magnitude(&self) -> f64 {
        f64::sqrt(norm_squared(self))
    }

    pub fn normalize(&mut self) {
        let m = self.magnitude();
        *self /= m;
    }
}

pub fn normalized(t: &Tuple4) -> Tuple4 {
    let mut new_t = t.clone();
    new_t.normalize();
    new_t
}

pub struct Ray {
    pub origin: Point,
    pub direction: Vector,
}

impl Ray {
    pub fn new(origin: Point, direction: Vector) -> Ray {
        Ray { origin, direction }
    }
}

pub fn position(ray: &Ray, t: f64) -> Point {
    &(ray.origin) + &(&ray.direction * t)
}

pub struct Sphere {
    pub origin: Point,
    pub radius: f64,
}

impl Sphere {
    pub fn new(origin: Point, radius: f64) -> Sphere {
        Sphere { origin, radius }
    }
}

pub fn instersect(s: Sphere, r: Ray) -> Vec<f64> {
    let sphere_to_ray = &r.origin - &s.origin;
    let a = dot(&r.direction, &r.direction);
    let b = 2.0 * dot(&r.direction, &sphere_to_ray);
    let c = dot(&sphere_to_ray, &sphere_to_ray) - s.radius;
    let delta = b * b - 4.0 * a * c;
    if delta < 0.0 {
        return vec![];
    }

    let t1 = (-b - f64::sqrt(delta)) / (2.0 * a);
    let t2 = (-b + f64::sqrt(delta)) / (2.0 * a);

    vec![t1, t2]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::float_compare::f64_approx_eq;

    #[test]
    fn adding_two_tuples() {
        let a1 = Tuple4::new([3.0, -2.0, 5.0, 1.0]);
        let a2 = Tuple4::new([-2.0, 3.0, 1.0, 0.0]);

        assert_eq!(&a1 + &a2, Tuple4::new([1.0, 1.0, 6.0, 1.0]));
        assert_eq!(a1.x() + a2.x(), 1.0); // check if not moved from
    }

    #[test]
    fn subtracting_two_points_gives_vector() {
        let p1 = point(3.0, 2.0, 1.0);
        let p2 = point(5.0, 6.0, 7.0);

        assert_eq!(&p1 - &p2, vector(-2.0, -4.0, -6.0));
        assert_eq!(p1.x() + p2.x(), 8.0); // check if not moved from
    }

    #[test]
    fn subtracting_vector_from_point() {
        let p = point(3.0, 2.0, 1.0);
        let v = vector(5.0, 6.0, 7.0);

        assert_eq!(&p - &v, point(-2.0, -4.0, -6.0));
        assert_eq!(p.x() + v.x(), 8.0); // check if not moved from
    }

    #[test]
    fn subtracting_two_vectors() {
        let p = vector(3.0, 2.0, 1.0);
        let v = vector(5.0, 6.0, 7.0);

        assert_eq!(&p - &v, vector(-2.0, -4.0, -6.0));
        assert_eq!(p.x() + v.x(), 8.0); // check if not moved from
    }

    #[test]
    fn subtracting_vector_from_zero() {
        let zero = vector(0.0, 0.0, 0.0);
        let v = vector(1.0, -2.0, 3.0);

        assert_eq!(&zero - &v, vector(-1.0, 2.0, -3.0));
        assert_eq!(zero.x() + v.x(), 1.0); // check if not moved from
    }

    #[test]
    fn negating_tuple() {
        let a = Tuple4::new([1.0, -2.0, 3.0, -4.0]);

        assert_eq!(-&a, Tuple4::new([-1.0, 2.0, -3.0, 4.0]));
        assert_eq!(a.x(), 1.0); // check if not moved from
    }

    #[test]
    fn multiply_tuple_by_scalar() {
        let a = Tuple4::new([1.0, -2.0, 3.0, -4.0]);

        assert_eq!(&a * 3.5, Tuple4::new([3.5, -7.0, 10.5, -14.0]));
        assert_eq!(a.x(), 1.0); // check if not moved from
    }

    #[test]
    fn multiply_tuple_by_fraction() {
        let a = Tuple4::new([1.0, -2.0, 3.0, -4.0]);

        assert_eq!(&a * 0.5, Tuple4::new([0.5, -1.0, 1.5, -2.0]));
        assert_eq!(a.x(), 1.0); // check if not moved from
    }

    #[test]
    fn dividing_tuple_by_scalar() {
        let a = Tuple4::new([1.0, -2.0, 3.0, -4.0]);

        assert_eq!(&a / 2.0, Tuple4::new([0.5, -1.0, 1.5, -2.0]));
        assert_eq!(a.x(), 1.0); // check if not moved from
    }

    #[test]
    fn magnitude_1_0_0_is_1() {
        let v = vector(1.0, 0.0, 0.0);
        let expected_magnitude = 1.0;

        assert!(f64_approx_eq(v.magnitude(), expected_magnitude));
        assert_eq!(v.x(), 1.0); // check if not moved from
    }

    #[test]
    fn magnitude_0_1_0_is_1() {
        let v = vector(0.0, 1.0, 0.0);
        let expected_magnitude = 1.0;

        assert!(f64_approx_eq(v.magnitude(), expected_magnitude));
        assert_eq!(v.x(), 0.0); // check if not moved from
    }

    #[test]
    fn magnitude_0_0_1_is_1() {
        let v = vector(0.0, 0.0, 1.0);
        let expected_magnitude = 1.0;

        assert!(f64_approx_eq(v.magnitude(), expected_magnitude));
        assert_eq!(v.x(), 0.0); // check if not moved from
    }

    #[test]
    fn magnitude_1_2_3_is_sqrt14() {
        let v = vector(1.0, 2.0, 3.0);
        let expected_magnitude = f64::sqrt(14.0);

        assert!(f64_approx_eq(v.magnitude(), expected_magnitude));
        assert_eq!(v.x(), 1.0); // check if not moved from
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

        assert_eq!(normalized(&v), vector(1.0, 0.0, 0.0));
        assert_eq!(v.x(), 4.0); // check if not moved from
    }

    #[test]
    fn normalizing_1_2_3_gives_stuff() {
        let v = vector(1.0, 2.0, 3.0);

        assert_eq!(
            normalized(&v),
            vector(0.267261241, 0.534522483, 0.801783725)
        );
        assert_eq!(v.x(), 1.0); // check if not moved from
    }

    #[test]
    fn normalized_vector_has_magnitude_1() {
        let v = vector(1.0, 2.0, 3.0);
        let normalized_v = normalized(&v);
        let expected_magnitude = 1.0;

        assert!(f64_approx_eq(normalized_v.magnitude(), expected_magnitude));
        assert_eq!(v.x(), 1.0); // check if not moved from
    }

    #[test]
    fn dot_product_of_tuples() {
        let a = vector(1.0, 2.0, 3.0);
        let b = vector(2.0, 3.0, 4.0);

        assert!(f64_approx_eq(dot(&a, &b), 20.0));
        assert!(f64_approx_eq(dot(&b, &a), 20.0))
    }

    #[test]
    fn cross_product_of_vectors() {
        let a = vector(1.0, 2.0, 3.0);
        let b = vector(2.0, 3.0, 4.0);

        assert_eq!(cross(&a, &b), vector(-1.0, 2.0, -1.0));
        assert_eq!(cross(&b, &a), vector(1.0, -2.0, 1.0));
    }

    #[test]
    fn multiply_matrix_by_vector() {
        let m = Matrix::<4, 4, _>::from_nested([
            [1, 2, 3, 4],
            [2, 4, 4, 2],
            [8, 6, 4, 1],
            [0, 0, 0, 1],
        ]);
        let v = Matrix::<4, 1, _>::from_nested([[1], [2], [3], [1]]);

        let expected = Matrix::<4, 1, _>::from_nested([[18], [24], [33], [1]]);

        assert_eq!(&m * &v, expected);
    }

    #[test]
    fn sphere_ray_two_point_intersect() {
        let r = Ray::new(point(0.0, 0.0, -5.0), vector(0.0, 0.0, 1.0));
        let s = Sphere::new(point(0.0, 0.0, 0.0), 1.0);
        let xs = instersect(s, r);
        assert_eq!(xs.len(), 2);
        assert!(xs[0].approx_eq(&4.0));
        assert!(xs[1].approx_eq(&6.0));
    }

    #[test]
    fn sphere_ray_one_point_intersect() {
        let r = Ray::new(point(0.0, 1.0, -5.0), vector(0.0, 0.0, 1.0));
        let s = Sphere::new(point(0.0, 0.0, 0.0), 1.0);
        let xs = instersect(s, r);
        assert_eq!(xs.len(), 2);
        assert!(xs[0].approx_eq(&5.0));
        assert!(xs[1].approx_eq(&5.0));
    }

    #[test]
    fn sphere_ray_no_point_of_intersect() {
        let r = Ray::new(point(0.0, 2.0, -5.0), vector(0.0, 0.0, 1.0));
        let s = Sphere::new(point(0.0, 0.0, 0.0), 1.0);
        let xs = instersect(s, r);
        assert_eq!(xs.len(), 0);
    }
}
