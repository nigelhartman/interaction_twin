#!/usr/bin/env python3
"""Original vector/Pillow explanatory figure; no model output is fabricated."""
import html
import json
from pathlib import Path
from PIL import Image, ImageDraw
from build_library_trailer_20261102 import OUT, BG, INK, MUTED, LIME, LINE, font, record, fresh


def main():
    directory = OUT / 'pipeline-v1'
    directory.mkdir()
    canvas = Image.new('RGB', (1800, 980), BG)
    draw = ImageDraw.Draw(canvas)
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="1800" height="980" viewBox="0 0 1800 980">',
           f'<rect width="1800" height="980" fill="{BG}"/>']
    def box(x, y, w, h, fill='white', stroke=LINE, dashed=False):
        draw.rounded_rectangle((x, y, x + w, y + h), 22, fill=fill, outline=stroke, width=2)
        svg.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="22" fill="{fill}" stroke="{stroke}" stroke-width="2"' + (' stroke-dasharray="8 7"' if dashed else '') + '/>')
    def label(x, y, value, size=26, color=INK, bold=False):
        draw.text((x, y), value, fill=color, font=font(size, bold))
        svg.append(f'<text x="{x}" y="{y + size * .91}" fill="{color}" font-family="Arial,sans-serif" font-size="{size}" font-weight="{700 if bold else 400}">{html.escape(value)}</text>')
    def arrow(points, color=MUTED):
        draw.line(points, fill=color, width=3, joint='curve')
        x, y = points[-1]
        draw.polygon([(x, y), (x - 12, y - 7), (x - 12, y + 7)], fill=color)
        svg.append('<polyline points="' + ' '.join(f'{x},{y}' for x, y in points) + f'" fill="none" stroke="{color}" stroke-width="3"/>')
        svg.append(f'<path d="M{x} {y}L{x - 12} {y - 7}L{x - 12} {y + 7}Z" fill="{color}"/>')
    label(65, 48, 'INTERACTION TWIN', 22, MUTED, True)
    label(62, 101, 'From demonstration to reusable interaction.', 52, INK, True)
    label(66, 178, 'Ordinary recordings become reviewable authoring material for people and agents.', 27, MUTED)
    box(64, 364, 250, 258)
    label(90, 395, '01 / CAPTURE', 18, MUTED, True)
    label(90, 444, 'Ordinary video', 29, INK, True)
    label(90, 499, 'Source RGB', 24, MUTED)
    label(90, 537, 'Exact timestamps', 24, MUTED)
    label(90, 575, 'No checkerboard', 24, MUTED)
    box(374, 280, 320, 186, '#e5efe8', '#9cb6a0')
    label(400, 307, '02 / PERCEPTION', 18, MUTED, True)
    label(400, 346, 'Meta SAM 3.1', 29, INK, True)
    label(400, 393, 'Visible parts + masks', 24, MUTED)
    label(400, 430, 'Source-bound proposals', 24, MUTED)
    box(374, 512, 320, 186, '#e5efe8', '#9cb6a0')
    label(400, 540, '03 / REASONING', 18, MUTED, True)
    label(400, 579, 'Muse Spark', 29, INK, True)
    label(400, 626, 'Relationships + states', 24, MUTED)
    label(400, 663, 'Steps + cited frames', 24, MUTED)
    arrow([(314, 494), (342, 494), (342, 372), (374, 372)])
    arrow([(342, 494), (342, 604), (374, 604)])
    box(758, 280, 340, 186, '#e6edf5', '#a3b5ca')
    label(784, 307, '04 / GEOMETRY', 18, MUTED, True)
    label(784, 346, 'Independent 3D drafts', 28, INK, True)
    label(784, 393, 'Source-derived surfaces', 24, MUTED)
    label(784, 430, 'Inferred models kept apart', 24, MUTED)
    box(758, 512, 340, 186, '#f4ebd7', '#ceb98f')
    label(784, 540, '05 / PROGRAM', 18, MUTED, True)
    label(784, 579, 'Descriptive interaction', 28, INK, True)
    label(784, 626, 'Components + actions', 24, MUTED)
    label(784, 663, 'Evidence + unknowns', 24, MUTED)
    arrow([(694, 372), (758, 372)])
    arrow([(694, 604), (758, 604)])
    box(1162, 364, 260, 258, INK, INK)
    label(1188, 395, '06 / LIBRARY', 18, '#bfd0b7', True)
    label(1188, 444, 'MCP + exports', 29, LIME, True)
    label(1188, 499, 'Find the interaction', 22, BG)
    label(1188, 537, 'Inspect the evidence', 22, BG)
    label(1188, 575, 'Retrieve GLB + JSON', 22, BG)
    arrow([(1098, 372), (1130, 372), (1130, 449), (1162, 449)])
    arrow([(1098, 604), (1130, 604), (1130, 546), (1162, 546)])
    box(1486, 364, 250, 258, '#f6f3e8', '#b8bfae', True)
    label(1512, 395, '07 / AUTHORING', 18, MUTED, True)
    label(1512, 444, 'Robotics + XR', 28, INK, True)
    label(1512, 499, 'Inspectable task draft', 21, MUTED)
    label(1512, 537, 'Illustrative rehearsal', 21, MUTED)
    label(1512, 575, 'Validation still needed', 21, MUTED)
    arrow([(1422, 493), (1486, 493)])
    box(64, 770, 1672, 140, '#e6ebdf', '#d5dece')
    label(90, 793, 'A REVIEWABLE DRAFT, WITH ITS LIMITS ATTACHED', 20, INK, True)
    label(90, 836, 'Shape, metric scale, common assembly, joints, collision and physical contact remain unverified.', 25, MUTED)
    label(90, 876, 'Illustrative pipeline diagram · actual saved stages · no robot policy or validated physics simulation claimed', 21, MUTED)
    svg.append('</svg>')
    vector = directory / 'interaction-pipeline.svg'
    vector.write_text('\n'.join(svg))
    raster = directory / 'interaction-pipeline.png'
    canvas.save(raster)
    fresh(directory / 'manifest.json', dict(kind='original_explanatory_pipeline_not_model_output',
        svg=record(vector, 'original_vector_schematic'), png=record(raster, 'same_layout_Pillow_schematic'),
        geometry_generated=False, published=False))
    print(json.dumps({'svg': str(vector), 'png': str(raster)}))


if __name__ == '__main__': main()
