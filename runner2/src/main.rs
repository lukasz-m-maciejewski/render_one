extern crate linear_algebra;

use linear_algebra::{cross, dot, vector};

fn main() {
    let v1 = vector(1.0, 2.0, 3.0);
    let v2 = vector(2.0, 1.0, -1.0);
    println!("runner 2 hello world {}", dot(&v1, &v2));
    println!("hello world {}", cross(&v1, &v2));
}
