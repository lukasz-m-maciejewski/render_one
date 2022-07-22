extern crate r1;

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

fn main() {
    let magnitude = std::env::args()
        .nth(1)
        .unwrap_or("1.0".to_string())
        .parse::<f64>()
        .unwrap();
    let env = Environment {
        gravity: vector(0.0, -0.1, 0.0),
        wind: vector(-0.01, 0.0, 0.0),
    };
    let mut p = Projectile {
        position: point(0.0, 1.0, 0.0),
        velocity: normalized(vector(1.0, 1.0, 0.0)) * magnitude,
    };

    let mut tick_count = 0;

    while p.position.y() > 0.0 {
        p = tick(&env, &p);
        tick_count += 1;
    }

    println!(
        "Projectile landed at {} with velocity {} after {tick_count} ticks.",
        p.position, p.velocity
    );
}
