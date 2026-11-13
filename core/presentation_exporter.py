"""PowerPoint presentation exporter using python-pptx with clean academic styling and speaker notes."""

import io
import re
from typing import List, Dict, Any
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE


# Design Tokens for Presentation
COLOR_BG = RGBColor(15, 23, 42)          # Deep Slate #0F172A
COLOR_CARD = RGBColor(30, 41, 59)        # Card #1E293B
COLOR_TITLE = RGBColor(248, 250, 252)    # White #F8FAFC
COLOR_TEXT = RGBColor(203, 213, 225)     # Slate text #CBD5E1
COLOR_ACCENT = RGBColor(56, 189, 248)    # Sky accent #38BDF8
COLOR_MUTED = RGBColor(148, 163, 184)    # Muted #94A3B8


def parse_slide_markdown(content: str) -> List[Dict[str, Any]]:
    """Parse slide generator output into structured slide dictionaries.

    Looks for sections demarcated by '### Slide' or '## Slide' or '---'.
    Extracts title, bullet points, speaker notes, and visual recommendations.
    """
    slides = []
    # Split content by Slide headers
    slide_chunks = re.split(r"(?:###|##)\s*Slide\s*\d*[:\s]*", content, flags=re.IGNORECASE)

    for chunk in slide_chunks:
        chunk = chunk.strip()
        if not chunk or len(chunk) < 20:
            continue

        lines = [line.strip() for line in chunk.split("\n") if line.strip()]
        if not lines:
            continue

        # First line is the title
        raw_title = lines[0].replace("**", "").replace("#", "").strip()
        if raw_title.lower().startswith("deck") or raw_title.lower().startswith("presentation"):
            continue

        bullets = []
        speaker_notes = ""
        visual_rec = ""
        current_section = "bullets"

        for line in lines[1:]:
            lower = line.lower()
            if "speaker notes" in lower:
                current_section = "notes"
                note_text = re.sub(r"(?i)\*\*speaker notes:?\*\*", "", line).strip()
                if note_text:
                    speaker_notes += note_text + " "
                continue
            elif "visual" in lower or "diagram" in lower or "recommendation" in lower:
                current_section = "visual"
                rec_text = re.sub(r"(?i)\*\*visual.*?:?\*\*", "", line).strip()
                if rec_text:
                    visual_rec += rec_text + " "
                continue

            if current_section == "bullets":
                cleaned_line = line.lstrip("*-•123456789. ").strip()
                cleaned_line = cleaned_line.replace("**", "").strip()
                if cleaned_line and not cleaned_line.lower().startswith("key points"):
                    bullets.append(cleaned_line)
            elif current_section == "notes":
                speaker_notes += line + " "
            elif current_section == "visual":
                visual_rec += line + " "

        if not bullets and not speaker_notes:
            continue

        slides.append({
            "title": raw_title,
            "bullets": bullets[:6],  # Max 6 bullets for slide legibility
            "speaker_notes": speaker_notes.strip(),
            "visual_recommendation": visual_rec.strip(),
        })

    # Fallback if markdown regex parsing didn't match
    if not slides:
        # Create a single summary slide from text
        slides.append({
            "title": "Research Presentation",
            "bullets": [line[:120] for line in content.split("\n") if line.strip().startswith(("-", "*", "•"))][:6] or ["Research overview based on uploaded paper."],
            "speaker_notes": "Refer to the generated markdown report for full textual analysis.",
            "visual_recommendation": "Conceptual diagram of system architecture.",
        })

    return slides


def generate_pptx_file(slides_data: List[Dict[str, Any]], paper_title: str = "Research Paper Presentation") -> io.BytesIO:
    """Create a styled 16:9 widescreen PowerPoint deck from parsed slides."""
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]

    for idx, slide_info in enumerate(slides_data, start=1):
        slide = prs.slides.add_slide(blank_layout)

        # 1. Dark Solid Background
        bg_shape = slide.shapes.add_shape(
            MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.333), Inches(7.5)
        )
        bg_shape.fill.solid()
        bg_shape.fill.fore_color.rgb = COLOR_BG
        bg_shape.line.fill.background()

        # 2. Header Slide Number & Tag
        tag_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.5), Inches(11.7), Inches(0.4))
        tag_tf = tag_box.text_frame
        tag_tf.word_wrap = True
        tag_p = tag_tf.paragraphs[0]
        tag_p.text = f"SLIDE {idx:02d} / {len(slides_data):02d}  ·  {paper_title[:55]}"
        tag_p.font.size = Pt(10)
        tag_p.font.bold = True
        tag_p.font.color.rgb = COLOR_ACCENT

        # 3. Slide Title
        title_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.9), Inches(11.7), Inches(0.9))
        title_tf = title_box.text_frame
        title_tf.word_wrap = True
        title_p = title_tf.paragraphs[0]
        title_p.text = slide_info["title"]
        title_p.font.size = Pt(26)
        title_p.font.bold = True
        title_p.font.color.rgb = COLOR_TITLE

        # 4. Content Card Container
        card = slide.shapes.add_shape(
            MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(2.0), Inches(11.733), Inches(4.8)
        )
        card.fill.solid()
        card.fill.fore_color.rgb = COLOR_CARD
        card.line.color.rgb = RGBColor(51, 65, 85)
        card.line.width = Pt(1)

        # 5. Bullet Points inside Content Card
        bullet_box = slide.shapes.add_textbox(Inches(1.1), Inches(2.2), Inches(11.1), Inches(3.6))
        bullet_tf = bullet_box.text_frame
        bullet_tf.word_wrap = True

        for b_idx, bullet in enumerate(slide_info.get("bullets", [])):
            p = bullet_tf.add_paragraph() if b_idx > 0 else bullet_tf.paragraphs[0]
            p.text = f"•  {bullet}"
            p.font.size = Pt(15)
            p.font.color.rgb = COLOR_TEXT
            p.space_after = Pt(10)

        # 6. Visual recommendation banner at the bottom of the card
        if slide_info.get("visual_recommendation"):
            rec_box = slide.shapes.add_textbox(Inches(1.1), Inches(5.9), Inches(11.1), Inches(0.7))
            rec_tf = rec_box.text_frame
            rec_tf.word_wrap = True
            rec_p = rec_tf.paragraphs[0]
            rec_p.text = f"Recommended Visual: {slide_info['visual_recommendation'][:140]}"
            rec_p.font.size = Pt(10.5)
            rec_p.font.italic = True
            rec_p.font.color.rgb = COLOR_MUTED

        # 7. Embed Speaker Notes into Presentation Notes Pane
        speaker_notes = slide_info.get("speaker_notes", "")
        if speaker_notes:
            notes_slide = slide.notes_slide
            notes_tf = notes_slide.notes_text_frame
            notes_tf.text = f"SPEAKER NOTES (Slide {idx}):\n\n{speaker_notes}"

    output_stream = io.BytesIO()
    prs.save(output_stream)
    output_stream.seek(0)
    return output_stream
