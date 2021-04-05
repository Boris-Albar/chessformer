/// Raw transmutation from `u32`.
///
/// Converts the given `u32` containing the float's raw memory representation into the `f32` type.
/// Similar to `f32::from_bits` but even more raw.
#[inline]
pub fn from_bits(x: u32) -> f32 {
    unsafe { ::std::mem::transmute::<u32, f32>(x) }
}

#[inline]
pub fn pow2(p: f32) -> f32 {
    let clipp = if p < -126.0 { -126.0_f32 } else { p };
    let v = ((1 << 23) as f32 * (clipp + 126.94269504_f32)) as u32;
    from_bits(v)
}

#[inline]
pub fn exp(p: f32) -> f32 {
    pow2(1.442695040_f32 * p)
}

/// Sigmoid function.
#[inline]
pub fn sigmoid(x: f32) -> f32 {
    1.0_f32 / (1.0_f32 + exp(-x))
}

#[inline]
pub fn tanh(p: f32) -> f32 {
    -1.0_f32 + 2.0_f32 / (1.0_f32 + exp(-2.0_f32 * p))
}
