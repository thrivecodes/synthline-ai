"""Interactive, standalone HTML visual dataset inspector."""

from __future__ import annotations

import base64
from pathlib import Path

import cv2
import numpy as np

from synthline_ai.generation.base import GenerationResult
from synthline_ai.labeling.masks import mask_area


def _array_to_base64_jpeg(image: np.ndarray, quality: int = 85) -> str:
    """Encode BGR numpy array to base64 JPEG data URL."""
    success, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not success:
        return ""
    b64_str = base64.b64encode(encoded.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{b64_str}"


def _create_overlay(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Create a semi-transparent red defect mask overlay on top of the image."""
    overlay = image.copy()
    defect_pixels = mask > 0
    if defect_pixels.any():
        # Red tint in BGR is (0, 0, 255)
        red_layer = np.zeros_like(image)
        red_layer[:, :, 2] = 255
        blended = cv2.addWeighted(overlay, 0.5, red_layer, 0.5, 0)
        overlay[defect_pixels] = blended[defect_pixels]
    return overlay


def generate_html_preview(
    results: list[GenerationResult],
    output_path: Path,
    title: str = "SynthLine AI — Dataset Preview",
    max_samples: int = 64,
) -> Path:
    """Generate a self-contained, interactive HTML preview gallery for a generation run.

    Args:
        results: List of GenerationResult objects.
        output_path: Path where the preview.html will be saved.
        title: Page title header.
        max_samples: Maximum number of sample cards to render.

    Returns:
        Path to the generated HTML file.
    """
    total_samples = len(results)
    sampled_results = results[:max_samples]

    defect_counts: dict[str, int] = {}
    split_counts: dict[str, int] = {}
    total_area_pct = 0.0

    cards_html_list: list[str] = []

    for idx, res in enumerate(sampled_results):
        defect = res.defect_type or "unknown"
        split = res.split or "train"
        defect_counts[defect] = defect_counts.get(defect, 0) + 1
        split_counts[split] = split_counts.get(split, 0) + 1

        img_h, img_w = res.image.shape[:2]
        area = mask_area(res.mask)
        area_pct = (area / max(img_h * img_w, 1)) * 100.0
        total_area_pct += area_pct

        img_b64 = _array_to_base64_jpeg(res.image)
        overlay = _create_overlay(res.image, res.mask)
        overlay_b64 = _array_to_base64_jpeg(overlay)

        card_html = f"""
        <div class="sample-card" data-defect="{defect}" data-split="{split}">
          <div class="image-wrapper">
            <img src="{img_b64}"
                 class="img-main"
                 data-original="{img_b64}"
                 data-overlay="{overlay_b64}"
                 alt="Generated Defect #{idx + 1}"
                 loading="lazy" />
          </div>
          <div class="card-details">
            <div class="card-header">
              <span class="sample-id">#{idx + 1:04d}</span>
              <span class="badge badge-defect">{defect}</span>
              <span class="badge badge-split">{split}</span>
            </div>
            <div class="card-meta">
              <span>Seed: <code>{res.source_seed or "synthetic"}</code></span>
              <span>Area: <strong>{area_pct:.2f}%</strong></span>
            </div>
          </div>
        </div>
        """
        cards_html_list.append(card_html)

    avg_area = total_area_pct / max(len(sampled_results), 1)
    all_cards = "\n".join(cards_html_list)

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{title}</title>
  <style>
    :root {{
      --color-page: #f5f4ef;
      --color-surface: #ffffff;
      --color-surface-subtle: #fbfaf6;
      --color-ink: #17212b;
      --color-ink-muted: #65717a;
      --color-line: #d8d8d0;
      --color-accent: #087f8c;
      --color-accent-soft: #e2f1f2;
      --radius: 8px;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      background-color: var(--color-page);
      color: var(--color-ink);
      line-height: 1.5;
      padding: 2rem;
    }}
    .container {{
      max-width: 1400px;
      margin: 0 auto;
    }}
    header {{
      background: var(--color-surface);
      border: 1px solid var(--color-line);
      border-radius: var(--radius);
      padding: 1.5rem 2rem;
      margin-bottom: 2rem;
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-wrap: wrap;
      gap: 1rem;
    }}
    h1 {{
      font-size: 1.5rem;
      font-weight: 700;
      color: var(--color-ink);
    }}
    .stats-bar {{
      display: flex;
      gap: 1.5rem;
      flex-wrap: wrap;
    }}
    .stat-item {{
      font-size: 0.875rem;
    }}
    .stat-label {{
      color: var(--color-ink-muted);
      text-transform: uppercase;
      font-size: 0.75rem;
      letter-spacing: 0.05em;
    }}
    .stat-value {{
      font-weight: 700;
      font-size: 1.125rem;
      display: block;
    }}
    .controls {{
      background: var(--color-surface);
      border: 1px solid var(--color-line);
      border-radius: var(--radius);
      padding: 1rem 1.5rem;
      margin-bottom: 2rem;
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-wrap: wrap;
      gap: 1rem;
    }}
    .btn-group {{
      display: inline-flex;
      background: var(--color-page);
      padding: 3px;
      border-radius: 6px;
      border: 1px solid var(--color-line);
    }}
    .btn {{
      border: none;
      background: transparent;
      padding: 6px 14px;
      font-size: 0.8125rem;
      font-weight: 600;
      color: var(--color-ink-muted);
      border-radius: 4px;
      cursor: pointer;
      transition: all 0.15s ease;
    }}
    .btn.active {{
      background: var(--color-surface);
      color: var(--color-ink);
      box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    }}
    .gallery-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
      gap: 1.5rem;
    }}
    .sample-card {{
      background: var(--color-surface);
      border: 1px solid var(--color-line);
      border-radius: var(--radius);
      overflow: hidden;
      display: flex;
      flex-direction: column;
      transition: transform 0.15s ease, box-shadow 0.15s ease;
    }}
    .sample-card:hover {{
      transform: translateY(-2px);
      box-shadow: 0 4px 12px rgba(0,0,0,0.06);
    }}
    .image-wrapper {{
      position: relative;
      background: #000;
      aspect-ratio: 1 / 1;
      display: flex;
      align-items: center;
      justify-content: center;
    }}
    .image-wrapper img {{
      width: 100%;
      height: 100%;
      object-fit: contain;
      display: block;
    }}
    .card-details {{
      padding: 1rem;
    }}
    .card-header {{
      display: flex;
      align-items: center;
      gap: 0.5rem;
      margin-bottom: 0.5rem;
    }}
    .sample-id {{
      font-family: monospace;
      font-size: 0.8125rem;
      font-weight: bold;
      color: var(--color-ink-muted);
    }}
    .badge {{
      font-size: 0.6875rem;
      padding: 2px 8px;
      border-radius: 999px;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }}
    .badge-defect {{
      background: var(--color-accent-soft);
      color: var(--color-accent);
    }}
    .badge-split {{
      background: var(--color-page);
      color: var(--color-ink-muted);
      border: 1px solid var(--color-line);
    }}
    .card-meta {{
      display: flex;
      justify-content: space-between;
      font-size: 0.8125rem;
      color: var(--color-ink-muted);
    }}
    code {{
      font-family: monospace;
      background: var(--color-page);
      padding: 2px 4px;
      border-radius: 3px;
    }}
  </style>
</head>
<body>
  <div class="container">
    <header>
      <div>
        <h1>{title}</h1>
        <p style="color: var(--color-ink-muted); font-size: 0.875rem;">
          Interactive visual dataset inspection gallery
        </p>
      </div>
      <div class="stats-bar">
        <div class="stat-item">
          <span class="stat-label">Total Generated</span>
          <span class="stat-value">{total_samples}</span>
        </div>
        <div class="stat-item">
          <span class="stat-label">Previewed Samples</span>
          <span class="stat-value">{len(sampled_results)}</span>
        </div>
        <div class="stat-item">
          <span class="stat-label">Avg Mask Area</span>
          <span class="stat-value">{avg_area:.2f}%</span>
        </div>
      </div>
    </header>

    <div class="controls">
      <div style="display: flex; align-items: center; gap: 0.75rem;">
        <span class="stat-label">View Mode:</span>
        <div class="btn-group" id="viewModeGroup">
          <button class="btn active" data-mode="generated">Generated</button>
          <button class="btn" data-mode="overlay">Mask Overlay</button>
        </div>
      </div>
      <div style="display: flex; align-items: center; gap: 0.75rem;">
        <span class="stat-label">Filter Split:</span>
        <div class="btn-group" id="splitFilterGroup">
          <button class="btn active" data-filter="all">All</button>
          <button class="btn" data-filter="train">Train</button>
          <button class="btn" data-filter="val">Val</button>
          <button class="btn" data-filter="test">Test</button>
        </div>
      </div>
    </div>

    <div class="gallery-grid" id="gallery">
      {all_cards}
    </div>
  </div>

  <script>
    // View Mode Toggle (Generated Image vs Mask Overlay)
    document.querySelectorAll('#viewModeGroup .btn').forEach(button => {{
      button.addEventListener('click', () => {{
        document.querySelectorAll('#viewModeGroup .btn').forEach(b => b.classList.remove('active'));
        button.classList.add('active');
        const mode = button.getAttribute('data-mode');
        document.querySelectorAll('.img-main').forEach(img => {{
          if (mode === 'overlay') {{
            img.src = img.getAttribute('data-overlay');
          }} else {{
            img.src = img.getAttribute('data-original');
          }}
        }});
      }});
    }});

    // Split Filter
    document.querySelectorAll('#splitFilterGroup .btn').forEach(button => {{
      button.addEventListener('click', () => {{
        const allBtns = document.querySelectorAll('#splitFilterGroup .btn');
        allBtns.forEach(b => b.classList.remove('active'));
        button.classList.add('active');
        const filter = button.getAttribute('data-filter');
        document.querySelectorAll('.sample-card').forEach(card => {{
          if (filter === 'all' || card.getAttribute('data-split') === filter) {{
            card.style.display = 'flex';
          }} else {{
            card.style.display = 'none';
          }}
        }});
      }});
    }});
  </script>
</body>
</html>
"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html_content, encoding="utf-8")
    return output_path
