extern crate string_builder;

use crate::color::Color;

use string_builder::Builder;

#[derive(Clone, Copy, Debug)]
pub struct Dimensions {
    pub width: usize,
    pub height: usize,
}

#[derive(Clone, Copy, Debug)]
pub struct Point {
    x: usize,
    y: usize,
}

#[derive(Clone, Debug)]
pub struct Canvas {
    dimensions: Dimensions,
    data: Vec<Color>,
}

impl Canvas {
    pub fn new(d: Dimensions) -> Canvas {
        Canvas {
            dimensions: d,
            data: vec![Color::new_rgb(0.0, 0.0, 0.0); d.width * d.height],
        }
    }

    fn point2idx(pos: Point, d: &Dimensions) -> usize {
        d.width * pos.y + pos.x
    }

    fn assert_is_inside(&self, pos: Point) {
        assert!(pos.x < self.dimensions.width && pos.y < self.dimensions.height);
    }

    pub fn width(&self) -> usize {
        self.dimensions.width
    }

    pub fn height(&self) -> usize {
        self.dimensions.height
    }

    pub fn write_pixel(&mut self, pos: Point, c: Color) {
        self.assert_is_inside(pos);
        self.data[Self::point2idx(pos, &self.dimensions)] = c;
    }

    pub fn pixel_at(&self, pos: Point) -> Color {
        self.assert_is_inside(pos);
        self.data[Self::point2idx(pos, &self.dimensions)]
    }

    pub fn raw_data(&self) -> &Vec<Color> {
        &self.data
    }
}

fn color_to_integer(value: f32) -> u8 {
   f32::ceil(255.0 * f32::clamp(value, 0.0, 1.0)) as u8
}

pub fn canvas_to_ppm(c: &Canvas) -> String {
    let mut builder = Builder::default();

    builder.append("P3\n");
    builder.append(format!("{} {}\n", c.width(), c.height()));
    builder.append("255\n");
    for pixel in c.raw_data() {
        builder.append(format!(
            "{} {} {}\n",
            color_to_integer(pixel.red()),
            color_to_integer(pixel.green()),
            color_to_integer(pixel.blue())
        ))
    }
    builder.string().unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canvas_writing_pixes() {
        let mut c = Canvas::new(Dimensions {
            width: 20,
            height: 10,
        });
        let red = Color::new_rgb(1.0, 0.0, 0.0);

        c.write_pixel(Point { x: 2, y: 3 }, red);
        assert_eq!(c.pixel_at(Point { x: 2, y: 3 }), red);
    }

    #[test]
    fn canvas_to_ppm_content() {
        let mut c = Canvas::new(Dimensions {
            width: 5,
            height: 3,
        });
        c.write_pixel(Point { x: 0, y: 0 }, Color::new_rgb(1.5, 0.0, 1.0));
        c.write_pixel(Point { x: 2, y: 1 }, Color::new_rgb(0.0, 0.5, 0.0));
        c.write_pixel(Point { x: 4, y: 2 }, Color::new_rgb(-0.5, 0.0, 1.0));
        let expected = r#"P3
5 3
255
255 0 255
0 0 0
0 0 0
0 0 0
0 0 0
0 0 0
0 0 0
0 128 0
0 0 0
0 0 0
0 0 0
0 0 0
0 0 0
0 0 0
0 0 255
"#;
        assert_eq!(canvas_to_ppm(&c), expected);
    }
}
