pub fn f32_approx_eq(a: f32, b: f32) -> bool {
    f32::abs(a - b) < 0.0001
}

pub fn f64_approx_eq(a: f64, b: f64) -> bool {
    f64::abs(a - b) < 0.00000001
}
