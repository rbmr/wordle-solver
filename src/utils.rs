use std::iter::Sum;
use std::ops::Mul;


const KB: f32 = 1024.0;
const MB: f32 = 1024.0 * KB;
const GB: f32 = 1024.0 * MB;

/// Formats a byte count into a human-readable string (KB, MB, GB).
pub fn format_bytes(bytes: f32) -> String {
    if bytes >= GB {
        format!("{:.2} GB", bytes / GB)
    } else if bytes >= MB {
        format!("{:.2} MB", bytes / MB)
    } else if bytes >= KB {
        format!("{:.2} KB", bytes / KB)
    } else {
        format!("{} bytes", bytes)
    }
}

#[inline]
pub fn sum_of_squares<T, I>(xs: I) -> T
where
    T: Mul<Output = T> + Sum + Copy,
    I: IntoIterator<Item = T>,
{
    xs.into_iter().map(|x| x * x).sum()
}
