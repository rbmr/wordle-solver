use plotters::prelude::*;
use std::path::Path;
use plotters::style::text_anchor::{HPos, Pos, VPos};
use plotters::prelude::SegmentValue;
use crate::sim::SimStats;

// --- Configuration ---
const CARD_WIDTH: u32 = 1000; // High resolution width
const CARD_HEIGHT: u32 = 500; // High resolution height
const FONT_FAMILY: &str = "sans-serif";

/// Main entry point to generate the comparison image.
/// Supports both .png and .svg extensions.
pub fn generate_comparison_image(
    mut results: Vec<(&str, SimStats)>,
    output_path: &Path
) -> Result<(), Box<dyn std::error::Error>> {
    // 1. Sort by descending total_guesses (highest/worst first, as requested)
    results.sort_by(|a, b| b.1.total_guesses().cmp(&a.1.total_guesses()));

    let n_plots = results.len() as u32;
    let dims = (CARD_WIDTH, n_plots * CARD_HEIGHT);
    let ext = output_path.extension().and_then(|s| s.to_str()).unwrap_or("png");

    // 2. Select Backend based on file extension
    if ext.eq_ignore_ascii_case("svg") {
        let root = SVGBackend::new(output_path, dims).into_drawing_area();
        root.fill(&WHITE)?;
        draw_all_cards(root, &results)?;
    } else {
        let root = BitMapBackend::new(output_path, dims).into_drawing_area();
        root.fill(&WHITE)?;
        draw_all_cards(root, &results)?;
    }

    Ok(())
}

/// Generic drawer that works for both SVG and Bitmap backends
fn draw_all_cards<DB: DrawingBackend>(
    root: DrawingArea<DB, plotters::coord::Shift>,
    results: &Vec<(&str, SimStats)>,
) -> Result<(), Box<dyn std::error::Error>>
where DB::ErrorType: 'static {

    // Global max for X-axis alignment
    let global_max_guesses = results.iter()
        .flat_map(|(_, s)| s.distribution.keys())
        .max()
        .copied()
        .unwrap_or(6) as u32;

    let chunks = root.split_evenly((results.len(), 1));

    for (idx, (label, stats)) in results.iter().enumerate() {
        draw_single_card(&chunks[idx], idx, label, stats, global_max_guesses)?;
    }

    root.present()?;
    Ok(())
}

fn draw_single_card<DB: DrawingBackend>(
    area: &DrawingArea<DB, plotters::coord::Shift>,
    color_idx: usize,
    label: &str,
    stats: &SimStats,
    max_x: u32,
) -> Result<(), Box<dyn std::error::Error>>
where DB::ErrorType: 'static {
    let area = area.margin(20, 20, 20, 20);

    let total_f64 = stats.total_words as f64;
    let max_count = *stats.distribution.values().max().unwrap_or(&0);

    // Y-Axis scale (Frequency)
    let max_pct = if stats.total_words > 0 {
        (max_count as f64 / total_f64) * 100.0
    } else {
        100.0
    };
    let y_axis_max = max_pct * 1.2; // 20% headroom

    // X-Axis scale (Segmented)
    let x_range = (1u32..(max_x + 1)).into_segmented();

    // --- Build Chart ---
    let mut chart = ChartBuilder::on(&area)
        .x_label_area_size(40)
        .y_label_area_size(60)
        // Leave space at the top for the overlay stats
        .margin_top(10)
        .build_cartesian_2d(x_range, 0f64..y_axis_max)?;

    chart.configure_mesh()
        .disable_x_mesh()
        .light_line_style(&WHITE.mix(0.8))
        .bold_line_style(&BLACK.mix(0.1))
        .y_desc("Frequency")
        .x_desc("# Guesses")
        .y_label_formatter(&|v| format!("{:.0}%", v))
        .x_label_style((FONT_FAMILY, 20).into_font())
        .y_label_style((FONT_FAMILY, 20).into_font())
        .axis_desc_style((FONT_FAMILY, 22).into_font().style(FontStyle::Bold))
        .draw()?;

    // --- Draw Bars ---
    let bar_color = get_palette_color(color_idx);

    chart.draw_series(
        Histogram::vertical(&chart)
            .style(bar_color.filled())
            .margin(12) // Spacing between bars
            .data(
                stats.distribution.iter().map(|(&guesses, &count)| {
                    (guesses as u32, (count as f64 / total_f64) * 100.0)
                })
            )
    )?;

    // --- Draw Labels on Bars ---
    // Create fonts first as FontDesc, then convert to TextStyle with Pos
    let pct_font = (FONT_FAMILY, 18).into_font().style(FontStyle::Bold).color(&BLACK);
    let count_font = (FONT_FAMILY, 16).into_font().color(&BLACK.mix(0.6));
    let pos = Pos::new(HPos::Center, VPos::Bottom);

    // We must convert FontDesc/TextStyle to style with pos for PointSeries
    let pct_style = pct_font.pos(pos);
    let count_style = count_font.pos(pos);

    chart.draw_series(PointSeries::of_element(
        stats.distribution.iter().map(|(&guesses, &count)| (guesses as u32, count)),
        5,
        &TRANSPARENT,
        &|coord, _size, _style| {
            let guesses = coord.0;
            let count = coord.1;
            let pct = (count as f64 / total_f64) * 100.0;
            let x_coord = SegmentValue::CenterOf(guesses);

            EmptyElement::at((x_coord, pct))
                + Text::new(format!("{:.1}%", pct), (0, -20), pct_style.clone())
                + Text::new(format!("({})", count), (0, -5), count_style.clone())
        }
    ))?;

    // --- Draw Stats Overlay (Top Right) ---
    // We calculate pixel coordinates relative to the drawing area size
    let (w, _h) = area.dim_in_pixel();

    // Define the stats content
    let mean = stats.mean();
    let var = stats.variance(mean);
    let (min, max) = stats.min_max();

    let table_data = vec![
        ("Mean", format!("{:.4}", mean)),
        ("Variance", format!("{:.4}", var)),
        ("Range", format!("{} - {}", min, max)),
        ("Samples", format!("{}", stats.total_words)),
    ];

    // Draw the stats block in the top right
    draw_stats_overlay(&area, w, label, table_data)?;

    Ok(())
}

/// Draws the Title and the Statistics Table aligned to the right
fn draw_stats_overlay<DB: DrawingBackend>(
    area: &DrawingArea<DB, plotters::coord::Shift>,
    area_width: u32,
    title: &str,
    data: Vec<(&str, String)>,
) -> Result<(), Box<dyn std::error::Error>>
where DB::ErrorType: 'static {

    // Definitions
    let title_font = (FONT_FAMILY, 42).into_font().style(FontStyle::Bold);
    let label_style_base = (FONT_FAMILY, 24).into_font().color(&BLACK.mix(0.7));
    let value_font = (FONT_FAMILY, 24).into_font().style(FontStyle::Bold);

    // 1. Measure Table Column Widths using area.estimate_text_size
    let mut max_label_w = 0;
    let mut max_val_w = 0;

    for (lbl, val) in &data {
        // Measure label (label_style_base is already a TextStyle)
        let (w, _) = area.estimate_text_size(lbl, &label_style_base).unwrap_or((0,0));
        if w > max_label_w { max_label_w = w; }

        // Measure value (must convert FontDesc to TextStyle)
        let val_style = TextStyle::from(value_font.clone());
        let (w, _) = area.estimate_text_size(val, &val_style).unwrap_or((0,0));
        if w > max_val_w { max_val_w = w; }
    }

    let col_gap = 20;

    // 2. Position Logic (Top Right)
    let right_margin = 30;
    let top_margin = 10;

    // Anchor X on the right side of the content
    let anchor_x = (area_width as i32) - (right_margin);
    let mut current_y = top_margin;

    // 3. Draw Title (Right Aligned)
    // FIX: Convert FontDesc to TextStyle explicitly to use .pos()
    let title_style = TextStyle::from(title_font.clone()).pos(Pos::new(HPos::Right, VPos::Top));
    area.draw_text(title, &title_style, (anchor_x, current_y))?;

    // Move down for table
    current_y += 40;

    // 4. Draw Table
    // FIX: Create styles with position anchors
    let label_style = label_style_base.pos(Pos::new(HPos::Right, VPos::Top));
    let value_style = TextStyle::from(value_font.clone()).pos(Pos::new(HPos::Center, VPos::Top));

    let label_end_x = anchor_x - (max_val_w as i32) - (col_gap);
    let value_center_x = anchor_x - (max_val_w as i32 / 2);

    for (lbl, val) in data {
        area.draw_text(lbl, &label_style, (label_end_x, current_y))?;
        area.draw_text(&val, &value_style, (value_center_x, current_y))?;
        current_y += 25; // Row height
    }

    Ok(())
}

fn get_palette_color(idx: usize) -> RGBColor {
    let palette = [
        RGBColor(65, 105, 225),  // Royal Blue
        RGBColor(220, 20, 60),   // Crimson
        RGBColor(34, 139, 34),   // Forest Green
        RGBColor(255, 140, 0),   // Dark Orange
        RGBColor(138, 43, 226),  // Blue Violet
        RGBColor(0, 128, 128),   // Teal
    ];
    palette[idx % palette.len()]
}