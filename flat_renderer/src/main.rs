extern crate r1;

use std::io::Write;

use r1::canvas::{canvas_to_ppm, Canvas};
use r1::color::Color;
use r1::linear_algebra::{point, vector};
use r1::ray_emitter::{Camera, RayEmitter};
use r1::sphere::{instersect, Sphere};
use r1::util::{Dimensions, PhysicalDimensions, Resolution};

fn main() {
    let sphere = Sphere::new(point(0.0, 0.0, 7.0), 2.0);
    let camera = Camera::new(
        point(0.0, 0.0, 0.0),
        vector(0.0, 0.0, 1.0),
        vector(0.0, 1.0, 0.0),
        3.0,
        Resolution {
            width: 1920,
            height: 1080,
        },
        PhysicalDimensions {
            width: 3.4,
            height: 2.0,
        },
    )
    .unwrap();

    let mut canvas = Canvas::new(Dimensions {
        width: 1920,
        height: 1080,
    });

    let black = Color::new_rgb(0.0, 0.0, 0.0);
    let red = Color::new_rgb(1.0, 0.0, 0.0);

    for eray in RayEmitter::new(camera) {
        let intersection = instersect(&sphere, &eray.ray);
        let color = if intersection.len() == 2 { red } else { black };
        canvas.write_pixel(eray.source, color)
    }

    let mut file = std::fs::File::create("the_render.ppm").expect("file create failed");
    file.write_all(canvas_to_ppm(&canvas).as_bytes())
        .expect("file write failed");
}
