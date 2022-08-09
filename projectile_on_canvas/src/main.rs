extern crate r1;

use std::io::Write;

use r1::canvas::{Canvas, Dimensions, ScreenPoint, canvas_to_ppm};
use r1::color::Color;
use r1::linear_algebra::{normalized, point, vector, Point, Vector};

struct Projectile {
    position: Point,
    velocity: Vector,
}

struct Environment {
    gravity: Vector,
    wind: Vector,
}

fn tick(env: &Environment, proj: &Projectile) -> Projectile {
    Projectile {
        position: proj.position + proj.velocity,
        velocity: proj.velocity + env.gravity + env.wind,
    }
}

fn point_to_screen(point: Point, screen_height: usize) -> ScreenPoint {
    ScreenPoint {
        x: point.x() as usize,
        y: screen_height - point.y() as usize,
    }
}

fn main() {
    let magnitude = std::env::args()
        .nth(1)
        .unwrap_or("11.25".to_string())
        .parse::<f64>()
        .unwrap();
    let env = Environment {
        gravity: vector(0.0, -0.1, 0.0),
        wind: vector(-0.01, 0.0, 0.0),
    };

    let mut p = Projectile {
        position: point(0.0, 1.0, 0.0),
        velocity: normalized(vector(1.0, 1.8, 0.0)) * magnitude,
    };

    let mut tick_count = 0;
    let mut canvas = Canvas::new(Dimensions {
        width: 900,
        height: 550,
    });
    let red = Color::new_rgb(1.0, 0.0, 0.0);

    while p.position.y() > 0.0 {
        canvas.write_pixel(point_to_screen(p.position, canvas.height()), red);
        p = tick(&env, &p);
        tick_count += 1;
    }

    println!(
        "Projectile landed at {} with velocity {} after {tick_count} ticks.",
        p.position, p.velocity
    );

    let mut file = std::fs::File::create("projectile_trace.ppm").expect("file create failed");
    file.write_all(canvas_to_ppm(&canvas).as_bytes()).expect("file write failed");
}
