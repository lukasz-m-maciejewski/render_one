use crate::{
    linear_algebra::*,
    matrix::dot,
    util::{PhysicalDimensions, Resolution, ScreenPoint},
};

#[derive(Clone, Debug, PartialEq)]
pub struct Camera {
    pub focus: Point,
    pub direction: Vector,
    pub up: Vector,
    pub lens_distance: f64,
    pub resolution: Resolution,
    pub physical_dimensions: PhysicalDimensions,
}

#[derive(Debug)]
pub enum CameraError {
    InvalidLensDistance,
    InvalidResolution,
    InvalidPhysicalDimensions,
}

impl std::error::Error for CameraError {}

impl std::fmt::Display for CameraError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            CameraError::InvalidLensDistance => write!(f, "CameraError::InvalidLensDistance"),
            CameraError::InvalidResolution => write!(f, "CameraError::InvalidResolution"),
            CameraError::InvalidPhysicalDimensions => {
                write!(f, "CameraError::InvalidPhysicalDimensions")
            }
        }
    }
}

impl Camera {
    pub fn new(
        focus: Point,
        direction: Vector,
        up: Vector,
        lens_distance: f64,
        resolution: Resolution,
        physical_dimensions: PhysicalDimensions,
    ) -> Result<Camera, CameraError> {
        if lens_distance <= 0.0 {
            return Err(CameraError::InvalidLensDistance);
        }
        if resolution.height == 0 || resolution.width == 0 {
            return Err(CameraError::InvalidResolution);
        }
        assert!(physical_dimensions.height > 0.0 && physical_dimensions.width > 0.0);

        let direction = normalized(&direction);
        let up = &up - &(&direction * dot(&up, &direction));
        let up = normalized(&up);

        Ok(Camera {
            focus,
            direction,
            up,
            lens_distance,
            resolution,
            physical_dimensions,
        })
    }
}

pub struct RayEmitter {
    camera: Camera,
    screen_point: ScreenPoint,
    top_left: Point,
    step_x: Vector,
    step_y: Vector,
}

impl RayEmitter {
    pub fn new(camera: Camera) -> RayEmitter {
        let (top_left, top_right, bottom_left) = Self::corners(&camera);
        let step_x = &(&top_right - &top_left) / (camera.resolution.width as f64);
        let step_y = &(&bottom_left - &top_left) / (camera.resolution.height as f64);
        RayEmitter {
            camera,
            screen_point: ScreenPoint { x: 0, y: 0 },
            top_left,
            step_x,
            step_y,
        }
    }

    fn corners(camera: &Camera) -> (Point, Point, Point) {
        todo!();
    }

    fn next_point(&self) -> Option<ScreenPoint> {
        if self.screen_point.x + 1 == self.camera.resolution.width
            && self.screen_point.y + 1 == self.camera.resolution.height
        {
            return None;
        }

        if self.screen_point.x + 1 == self.camera.resolution.width {
            return Some(ScreenPoint {
                x: 0,
                y: self.screen_point.y + 1,
            });
        }

        Some(ScreenPoint {
            x: self.screen_point.x + 1,
            y: self.screen_point.y,
        })
    }
}

impl Iterator for RayEmitter {
    type Item = Ray;
    fn next(&mut self) -> Option<Self::Item> {
        if let Some(screen_point) = self.next_point() {
            self.screen_point = screen_point;
            let pos3d = &(&self.top_left + &(&self.step_x * (screen_point.x as f64)))
                + &(&self.step_y * (screen_point.y as f64));
            Some(Ray {
                origin: self.camera.focus.clone(),
                direction: normalized(&(&pos3d - &self.camera.focus)),
            })
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn camera_creation_no_corrections() {
        let camera = Camera::new(
            point(0.0, 0.0, 0.5),
            vector(0.0, 0.0, 1.0),
            vector(0.0, 1.0, 0.0),
            1.0,
            Resolution {
                width: 100,
                height: 100,
            },
            PhysicalDimensions {
                width: 10.0,
                height: 10.0,
            },
        )
        .unwrap();

        assert_eq!(camera.focus, point(0.0, 0.0, 0.5));
        assert_eq!(camera.direction, vector(0.0, 0.0, 1.0));
        assert_eq!(camera.up, vector(0.0, 1.0, 0.0));
    }

    #[test]
    fn camera_creation_up_corrected() {
        let camera = Camera::new(
            point(0.0, 0.0, 0.5),
            vector(0.0, 0.0, 1.0),
            vector(0.0, 1.0, 1.0),
            1.0,
            Resolution {
                width: 100,
                height: 100,
            },
            PhysicalDimensions {
                width: 10.0,
                height: 10.0,
            },
        )
        .unwrap();

        assert_eq!(camera.focus, point(0.0, 0.0, 0.5));
        assert_eq!(camera.direction, vector(0.0, 0.0, 1.0));
        assert_eq!(camera.up, vector(0.0, 1.0, 0.0));
    }

    #[test]
    fn camera_creation_direction_normalized() {
        let camera = Camera::new(
            point(0.0, 0.0, 0.5),
            vector(0.0, 0.0, 3.0),
            vector(0.0, 1.0, 0.0),
            1.0,
            Resolution {
                width: 100,
                height: 100,
            },
            PhysicalDimensions {
                width: 10.0,
                height: 10.0,
            },
        )
        .unwrap();

        assert_eq!(camera.focus, point(0.0, 0.0, 0.5));
        assert_eq!(camera.direction, vector(0.0, 0.0, 1.0));
        assert_eq!(camera.up, vector(0.0, 1.0, 0.0));
    }
}