#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Dimensions {
    pub width: usize,
    pub height: usize,
}

pub type Resolution = Dimensions;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PhysicalDimensions {
    pub width: f64,
    pub height: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ScreenPoint {
    pub x: usize,
    pub y: usize,
}
