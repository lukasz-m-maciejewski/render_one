use crate::float_compare::f32_approx_eq;

#[derive(Clone, Copy, Debug)]
pub struct Color {
    data: [f32; 3],
}

impl Color {
    pub fn new_rgb(red: f32, green: f32, blue: f32) -> Color {
        Color {
            data: [red, green, blue],
        }
    }

    pub fn red(&self) -> f32 {
        self.data[0]
    }

    pub fn green(&self) -> f32 {
        self.data[1]
    }

    pub fn blue(&self) -> f32 {
        self.data[2]
    }
}

impl std::cmp::PartialEq for Color {
    fn eq(&self, other: &Self) -> bool {
        std::iter::zip(self.data, other.data).all(|(l, r)| f32_approx_eq(l, r))
    }
}

impl std::cmp::Eq for Color {}

impl std::ops::Add for Color {
    type Output = Color;
    fn add(self, other: Color) -> Self::Output {
        let d: Vec<f32> = std::iter::zip(self.data, other.data)
            .map(|(l, r)| l + r)
            .collect();
        Color {
            data: d.try_into().unwrap(),
        }
    }
}

impl std::ops::Sub for Color {
    type Output = Color;
    fn sub(self, other: Color) -> Self::Output {
        let d: Vec<f32> = std::iter::zip(self.data, other.data)
            .map(|(l, r)| l - r)
            .collect();
        Color {
            data: d.try_into().unwrap(),
        }
    }
}

impl std::ops::Mul<f32> for Color {
    type Output = Color;
    fn mul(self, factor: f32) -> Self::Output {
        let mut new_data = self.data.clone();
        new_data
            .iter_mut()
            .for_each(|component| *component *= factor);
        Color { data: new_data }
    }
}

impl std::ops::Mul for Color {
    type Output = Color;
    fn mul(self, other: Color) -> Self::Output {
        let d: Vec<f32> = std::iter::zip(self.data, other.data)
            .map(|(l, r)| l * r)
            .collect();
        Color {
            data: d.try_into().unwrap(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::float_compare::f32_approx_eq;

    #[test]
    fn color_construction() {
        let c = Color::new_rgb(-0.5, 0.4, 1.7);

        assert!(f32_approx_eq(c.red(), -0.5));
        assert!(f32_approx_eq(c.green(), 0.4));
        assert!(f32_approx_eq(c.blue(), 1.7));
    }

    #[test]
    fn color_addition() {
        let c1 = Color::new_rgb(0.9, 0.6, 0.75);
        let c2 = Color::new_rgb(0.7, 0.1, 0.25);

        assert_eq!(c1 + c2, Color::new_rgb(1.6, 0.7, 1.0));
    }

    #[test]
    fn color_subtraction() {
        let c1 = Color::new_rgb(0.9, 0.6, 0.75);
        let c2 = Color::new_rgb(0.7, 0.1, 0.25);

        assert_eq!(c1 - c2, Color::new_rgb(0.2, 0.5, 0.5));
    }

    #[test]
    fn color_multiply_by_scalar() {
        let c = Color::new_rgb(0.2, 0.3, 0.4);

        assert_eq!(c * 2.0, Color::new_rgb(0.4, 0.6, 0.8));
    }

    #[test]
    fn color_multiply_pointwise() {
        let c1 = Color::new_rgb(1.0, 0.2, 0.4);
        let c2 = Color::new_rgb(0.9, 1.0, 0.1);

        assert_eq!(c1 * c2, Color::new_rgb(0.9, 0.2, 0.04));
    }
}
