use crate::{linear_algebra::*, matrix::*, ray::Ray};

#[derive(Clone, Debug, PartialEq)]
pub struct Sphere {
    pub origin: Point,
    pub radius: f64,
}

impl Sphere {
    pub fn new(origin: Point, radius: f64) -> Sphere {
        Sphere { origin, radius }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Intersection {
    pub t: f64,
    pub object: Sphere,
}

impl Intersection {
    pub fn new(t: f64, object: Sphere) -> Intersection {
        Intersection { t, object }
    }
}

pub fn instersect(s: &Sphere, r: &Ray) -> Vec<Intersection> {
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

    vec![
        Intersection::new(t1, s.clone()),
        Intersection::new(t2, s.clone()),
    ]
}

pub fn hit(is: &Vec<Intersection>) -> Option<Intersection> {
    if let Option::Some(t) = is
        .iter()
        .filter(|&i| i.t >= 0.0)
        .min_by(|&x, &y| x.t.total_cmp(&y.t))
    {
        return Option::Some(t.clone());
    } else {
        return Option::None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sphere_ray_two_point_intersect() {
        let r = Ray::new(point(0.0, 0.0, -5.0), vector(0.0, 0.0, 1.0));
        let s = Sphere::new(point(0.0, 0.0, 0.0), 1.0);
        let xs = instersect(&s, &r);
        assert_eq!(xs.len(), 2);
        assert!(xs[0].t.approx_eq(&4.0));
        assert!(xs[1].t.approx_eq(&6.0));
    }

    #[test]
    fn sphere_ray_one_point_intersect() {
        let r = Ray::new(point(0.0, 1.0, -5.0), vector(0.0, 0.0, 1.0));
        let s = Sphere::new(point(0.0, 0.0, 0.0), 1.0);
        let xs = instersect(&s, &r);
        assert_eq!(xs.len(), 2);
        assert!(xs[0].t.approx_eq(&5.0));
        assert!(xs[1].t.approx_eq(&5.0));
    }

    #[test]
    fn sphere_ray_no_point_of_intersect() {
        let r = Ray::new(point(0.0, 2.0, -5.0), vector(0.0, 0.0, 1.0));
        let s = Sphere::new(point(0.0, 0.0, 0.0), 1.0);
        let xs = instersect(&s, &r);
        assert_eq!(xs.len(), 0);
    }

    #[test]
    fn sphere_ray_inside() {
        let r = Ray::new(point(0.0, 0.0, 0.0), vector(0.0, 0.0, 1.0));
        let s = Sphere::new(point(0.0, 0.0, 0.0), 1.0);
        let xs = instersect(&s, &r);
        assert_eq!(xs.len(), 2);
        assert!(xs[0].t.approx_eq(&-1.0));
        assert!(xs[1].t.approx_eq(&1.0));
    }

    #[test]
    fn sphere_ray_points_away() {
        let r = Ray::new(point(0.0, 0.0, 5.0), vector(0.0, 0.0, 1.0));
        let s = Sphere::new(point(0.0, 0.0, 0.0), 1.0);
        let xs = instersect(&s, &r);
        assert_eq!(xs.len(), 2);
        assert!(xs[0].t.approx_eq(&-6.0));
        assert!(xs[1].t.approx_eq(&-4.0));
    }

    #[test]
    fn sphere_ray_two_point_intersect_with_expected_object() {
        let r = Ray::new(point(0.0, 0.0, -5.0), vector(0.0, 0.0, 1.0));
        let s = Sphere::new(point(0.0, 0.0, 0.0), 1.0);
        let xs = instersect(&s, &r);
        assert_eq!(xs.len(), 2);
        assert_eq!(xs[0].object, s);
        assert_eq!(xs[1].object, s);
    }

    #[test]
    fn hit_all_isect_positive_t() {
        let s = Sphere::new(point(0.0, 0.0, 0.0), 1.0);
        let xs = vec![
            Intersection::new(1.0, s.clone()),
            Intersection::new(2.0, s.clone()),
        ];

        assert_eq!(hit(&xs).unwrap(), xs[0]);
    }

    #[test]
    fn hit_some_isect_negative_t() {
        let s = Sphere::new(point(0.0, 0.0, 0.0), 1.0);
        let xs = vec![
            Intersection::new(1.0, s.clone()),
            Intersection::new(-1.0, s.clone()),
        ];

        assert_eq!(hit(&xs).unwrap(), xs[0]);
    }

    #[test]
    fn hit_all_isect_negative_t() {
        let s = Sphere::new(point(0.0, 0.0, 0.0), 1.0);
        let xs = vec![
            Intersection::new(-1.0, s.clone()),
            Intersection::new(-2.0, s.clone()),
        ];

        assert_eq!(hit(&xs), Option::None);
    }

    #[test]
    fn hit_multiple_isect_mixed_t() {
        let s = Sphere::new(point(0.0, 0.0, 0.0), 1.0);
        let xs = vec![
            Intersection::new(5.0, s.clone()),
            Intersection::new(7.0, s.clone()),
            Intersection::new(-3.0, s.clone()),
            Intersection::new(2.0, s.clone()),
        ];

        assert_eq!(hit(&xs).unwrap(), xs[3]);
    }
}
