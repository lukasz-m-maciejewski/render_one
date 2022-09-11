use crate::linear_algebra::*;
use crate::transformations::*;

#[derive(Clone, Debug, PartialEq)]
pub struct Ray {
    pub origin: Point,
    pub direction: Vector,
}

impl Ray {
    pub fn new(origin: Point, direction: Vector) -> Ray {
        Ray { origin, direction }
    }

    pub fn transform(&self, _transformation: &Transformation) -> Ray {
        self.clone()
    }
}

pub fn position(ray: &Ray, t: f64) -> Point {
    &(ray.origin) + &(&ray.direction * t)
}
