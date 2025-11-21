// notice: this code is almost entirely llm generated
// making plots is boring and easy

use plotters::prelude::*;
use std::path::Path;
use plotters::style::text_anchor::{HPos, Pos, VPos};
use plotters::prelude::SegmentValue;
use crate::game::sim::SimStats;

const CANVAS_WIDTH: u32 = 800;
const ROW_HEIGHT: u32 = 300;
const FONT_FAMILY: &str = "sans-serif";

pub fn generate_comparison_image(
    results: Vec<(&str, SimStats)>,
    output_path: &Path
) -> Result<(), Box<dyn std::error::Error>> {
    let n_plots = results.len() as u32;
    let canvas_height = n_plots * ROW_HEIGHT;

    // 1. Calculate Global Max Guesses
    let global_max_guesses = results.iter()
        .flat_map(|(_, s)| s.distribution.keys())
        .max()
        .copied()
        .unwrap_or(6) as u32;

    let root = BitMapBackend::new(output_path, (CANVAS_WIDTH, canvas_height))
        .into_drawing_area();

    root.fill(&WHITE)?;

    let chunks = root.split_evenly((n_plots as usize, 1));

    for (idx, (label, stats)) in results.into_iter().enumerate() {
        draw_single_heuristic(&chunks[idx], label, &stats, global_max_guesses)?;
    }

    root.present()?;
    Ok(())
}

fn draw_single_heuristic(
    root: &DrawingArea<BitMapBackend, plotters::coord::Shift>,
    label: &str,
    stats: &SimStats,
    global_max_guesses: u32,
) -> Result<(), Box<dyn std::error::Error>> {
    let split = root.split_horizontally(300);
    let left_panel = &split.0;
    let right_panel = &split.1;

    // --- Draw Stats Panel ---
    let title_style = TextStyle::from((FONT_FAMILY, 25).into_font()).color(&BLACK);
    let text_style = TextStyle::from((FONT_FAMILY, 18).into_font()).color(&BLACK);

    left_panel.draw_text(label, &title_style, (20, 30))?;

    let mean = stats.mean();
    let variance = stats.variance(mean);
    let (min, max) = stats.min_max();

    let lines = vec![
        format!("Mean Guesses: {:.4}", mean),
        format!("Variance: {:.4}", variance),
        format!("Min Guesses: {}", min),
        format!("Max Guesses: {}", max),
        format!("Total Words: {}", stats.total_words),
    ];

    for (i, line) in lines.iter().enumerate() {
        left_panel.draw_text(line, &text_style, (20, 80 + (i as i32 * 30)))?;
    }

    // --- Draw Histogram Panel ---
    let total_f64 = stats.total_words as f64;
    let max_count = *stats.distribution.values().max().unwrap_or(&0);
    let max_pct = if stats.total_words > 0 {
        (max_count as f64 / total_f64) * 100.0
    } else {
        100.0
    };

    let y_axis_max = max_pct * 1.2;

    // Use segmented coords for X.
    let x_range = (1u32..(global_max_guesses + 1)).into_segmented();

    let mut chart = ChartBuilder::on(right_panel)
        .margin(20)
        .x_label_area_size(30)
        .y_label_area_size(50)
        .build_cartesian_2d(x_range, 0f64..y_axis_max)?;

    chart.configure_mesh()
        .disable_x_mesh()
        .bold_line_style(&WHITE.mix(0.3))
        .x_desc("Number of Guesses")
        .y_desc("Percentage")
        .y_label_formatter(&|v| format!("{:.0}%", v))
        .axis_desc_style((FONT_FAMILY, 15))
        .draw()?;

    // PASS 1: Draw the Bars
    chart.draw_series(
        Histogram::vertical(&chart)
            .style(RGBColor(106, 170, 100).filled())
            .margin(5)
            .data(
                stats.distribution.iter().map(|(&guesses, &count)| {
                    (guesses as u32, (count as f64 / total_f64) * 100.0)
                })
            )
    )?;

    // PASS 2: Draw the Labels
    // Define the shared text style with the anchor applied
    let label_style = (FONT_FAMILY, 11)
        .into_font()
        .color(&BLACK)
        .pos(Pos::new(HPos::Center, VPos::Bottom)); // Anchor applied to the style, not the Element

    chart.draw_series(PointSeries::of_element(
        stats.distribution.iter().map(|(&guesses, &count)| (guesses as u32, count as u32)),
        5,
        &TRANSPARENT,
        &|coord, _size, _style| {
            let guesses = coord.0;
            let count = coord.1;
            let pct = (count as f64 / total_f64) * 100.0;

            // We must wrap the 'u32' guess in SegmentValue::CenterOf
            // to match the segmented coordinate system of the chart.
            let x_coord = SegmentValue::CenterOf(guesses);

            EmptyElement::at((x_coord, pct))
                + Text::new(
                    format!("({:.1}%)", pct),
                    (0, -5),
                    label_style.clone()
                )
                + Text::new(
                    format!("{}", count),
                    (0, -20),
                    label_style.clone().transform(FontTransform::None) // Ensure distinct object if needed, usually clone is enough
                )
        }
    ))?;

    Ok(())
}