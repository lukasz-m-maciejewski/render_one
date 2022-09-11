use crate::matrix::*;

pub type Transformation = Matrix<4, 4, f64>;

pub fn translation(x: f64, y: f64, z: f64) -> Transformation {
    let mut t = identity::<4, f64>();
    t[(0, 3)] = x;
    t[(1, 3)] = y;
    t[(2, 3)] = z;
    t
}

pub fn scaling(x: f64, y: f64, z: f64) -> Transformation {
    let mut t = identity::<4, f64>();
    t[(0, 0)] = x;
    t[(1, 1)] = y;
    t[(2, 2)] = z;
    t
}

pub fn rotation_x(angle: f64) -> Transformation {
    let mut t = identity::<4, f64>();
    t[(1, 1)] = f64::cos(angle);
    t[(1, 2)] = -f64::sin(angle);
    t[(2, 1)] = f64::sin(angle);
    t[(2, 2)] = f64::cos(angle);
    t
}

pub fn rotation_y(angle: f64) -> Transformation {
    let mut t = identity::<4, f64>();
    t[(0, 0)] = f64::cos(angle);
    t[(0, 2)] = f64::sin(angle);
    t[(2, 0)] = -f64::sin(angle);
    t[(2, 2)] = f64::cos(angle);
    t
}

pub fn rotation_z(angle: f64) -> Transformation {
    let mut t = identity::<4, f64>();
    t[(0, 0)] = f64::cos(angle);
    t[(0, 1)] = -f64::sin(angle);
    t[(1, 0)] = f64::sin(angle);
    t[(1, 1)] = f64::cos(angle);
    t
}

pub fn shearing(x_y: f64, x_z: f64, y_x: f64, y_z: f64, z_x: f64, z_y: f64) -> Transformation {
    let mut t = identity::<4, f64>();
    t[(0, 1)] = x_y;
    t[(0, 2)] = x_z;
    t[(1, 0)] = y_x;
    t[(1, 2)] = y_z;
    t[(2, 0)] = z_x;
    t[(2, 1)] = z_y;
    t
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linear_algebra::*;

    #[test]
    fn translation_test_1() {
        let p = point(-3.0, 4.0, 5.0);
        let transform = translation(5.0, -3.0, 2.0);

        assert_eq!(&transform * &p, point(2.0, 1.0, 7.0));
    }

    #[test]
    fn translation_inverse_test() {
        let transform = translation(5., -3., 2.);
        let inv = inverse_matrix(&transform).unwrap();
        let p = point(-3., 4., 5.);

        assert_eq!(&inv * &p, point(-8., 7., 3.));
    }

    #[test]
    fn translation_constant_on_vectors() {
        let transform = translation(5., -3., 2.);
        let v = vector(-3., 4., 5.);

        assert_eq!(&transform * &v, v);
    }

    #[test]
    fn scaling_point() {
        let transform = scaling(2., 3., 4.);
        let p = point(-4., 6., 8.);

        assert_eq!(&transform * &p, point(-8., 18., 32.));
    }

    #[test]
    fn scaling_vector() {
        let transform = scaling(2., 3., 4.);
        let v = vector(-4., 6., 8.);

        assert_eq!(&transform * &v, vector(-8., 18., 32.));
    }

    #[test]
    fn scaling_vector_by_inverse() {
        let transform = scaling(2., 3., 4.);
        let inv = inverse_matrix(&transform).unwrap();
        let v = vector(-4., 6., 8.);

        assert_eq!(&inv * &v, vector(-2., 2., 2.));
    }

    #[test]
    fn scaling_point_via_negative_is_reflection() {
        let transform = scaling(-1., 1., 1.);
        let p = point(2., 3., 4.);

        assert_eq!(&transform * &p, point(-2., 3., 4.));
    }

    #[test]
    fn rotation_around_x_axis() {
        let p = point(0.0, 1.0, 0.0);
        let half_quarter = rotation_x(std::f64::consts::FRAC_PI_4);
        let full_quarter = rotation_x(std::f64::consts::FRAC_PI_2);

        let sqrt_2_over_2 = std::f64::consts::SQRT_2 / 2.0;
        assert_eq!(&half_quarter * &p, point(0.0, sqrt_2_over_2, sqrt_2_over_2));
        assert_eq!(&full_quarter * &p, point(0.0, 0.0, 1.0));
    }

    #[test]
    fn rotation_around_x_axis_inverted() {
        let p = point(0.0, 1.0, 0.0);
        let half_quarter = rotation_x(std::f64::consts::FRAC_PI_4);
        let inv = inverse_matrix(&half_quarter).unwrap();

        let sqrt_2_over_2 = std::f64::consts::SQRT_2 / 2.0;
        assert_eq!(&inv * &p, point(0.0, sqrt_2_over_2, -sqrt_2_over_2));
    }

    #[test]
    fn rotation_around_y_axis() {
        let p = point(0.0, 0.0, 1.0);
        let half_quarter = rotation_y(std::f64::consts::FRAC_PI_4);
        let full_quarter = rotation_y(std::f64::consts::FRAC_PI_2);

        let sqrt_2_over_2 = std::f64::consts::SQRT_2 / 2.0;
        assert_eq!(&half_quarter * &p, point(sqrt_2_over_2, 0.0, sqrt_2_over_2));
        assert_eq!(&full_quarter * &p, point(1.0, 0.0, 0.0));
    }

    #[test]
    fn rotation_around_z_axis() {
        let p = point(0.0, 1.0, 0.0);
        let half_quarter = rotation_z(std::f64::consts::FRAC_PI_4);
        let full_quarter = rotation_z(std::f64::consts::FRAC_PI_2);

        let sqrt_2_over_2 = std::f64::consts::SQRT_2 / 2.0;
        assert_eq!(
            &half_quarter * &p,
            point(-sqrt_2_over_2, sqrt_2_over_2, 0.0)
        );
        assert_eq!(&full_quarter * &p, point(-1.0, 0.0, 0.0));
    }

    #[test]
    fn shearing_x_in_y() {
        let transform = shearing(1.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        let p = point(2.0, 3.0, 4.0);

        assert_eq!(&transform * &p, point(5.0, 3.0, 4.0));
    }

    #[test]
    fn shearing_x_in_z() {
        let transform = shearing(0.0, 1.0, 0.0, 0.0, 0.0, 0.0);
        let p = point(2.0, 3.0, 4.0);

        assert_eq!(&transform * &p, point(6.0, 3.0, 4.0));
    }

    #[test]
    fn shearing_y_in_x() {
        let transform = shearing(0.0, 0.0, 1.0, 0.0, 0.0, 0.0);
        let p = point(2.0, 3.0, 4.0);

        assert_eq!(&transform * &p, point(2.0, 5.0, 4.0));
    }

    #[test]
    fn shearing_y_in_z() {
        let transform = shearing(0.0, 0.0, 0.0, 1.0, 0.0, 0.0);
        let p = point(2.0, 3.0, 4.0);

        assert_eq!(&transform * &p, point(2.0, 7.0, 4.0));
    }

    #[test]
    fn shearing_z_in_x() {
        let transform = shearing(0.0, 0.0, 0.0, 0.0, 1.0, 0.0);
        let p = point(2.0, 3.0, 4.0);

        assert_eq!(&transform * &p, point(2.0, 3.0, 6.0));
    }

    #[test]
    fn shearing_z_in_y() {
        let transform = shearing(0.0, 0.0, 0.0, 0.0, 0.0, 1.0);
        let p = point(2.0, 3.0, 4.0);

        assert_eq!(&transform * &p, point(2.0, 3.0, 7.0));
    }
}
