"""Brief rendering: markdown + agency PDF (region-aware)."""
from __future__ import annotations
from .schema import ProductField, region_hint, REGION_RULES


def render_markdown(state: dict[str, ProductField], product_title: str = "Packaging Brief") -> str:
    def v(k: str) -> str:
        pf = state.get(k)
        if not pf or not pf.value:
            return "_— missing, see Open Questions —_"
        cites = " ".join(f"`[{s.file} {s.loc}]`" for s in pf.sources[:2])
        flag = " ⚠️" if pf.status in ("conflict", "needs_review") else ""
        return f"{pf.value}{flag} {cites}".strip()

    region = region_hint(state)
    rule = REGION_RULES.get(region, "")
    return f"""# {product_title} — Design Agency Brief

_Region: {region}. Printer rule: {rule}._

## 1. Product Overview
- **Product:** {v('product_name')}
- **Category:** {v('product_category')} | **Market:** {v('target_market')}
- **Pack format:** {v('pack_format')} | **Size:** {v('pack_size')} | **Net qty:** {v('net_quantity')}
- **Serving:** {v('serving_size')} | **Servings/pack:** {v('servings_per_pack')}

## 2. Pack Copy (must appear correctly)
- **Ingredients:** {v('ingredients')}
- **Allergens:** {v('allergens')}
- **Nutrition:** {v('nutrition')}
- **Claims:** {v('claims')} | **Certifications:** {v('certifications')}

## 3. Branding Rules
- **Logo:** {v('branding_logo')}
- **Tagline:** {v('tagline')}
- **Design notes:** {v('design_notes')} | **Languages:** {v('languages')}

## 4. Mandatory Content (printer must include)
{v('mandatory_content')}

## 5. Open Questions for Agency
_Fields marked missing/conflict above must be confirmed before artwork starts. Do not guess allergens, net quantity, serving size, or {region} marks._
"""


def export_pdf(markdown_text: str, out_path: str) -> str:
    from fpdf import FPDF
    pdf = FPDF(format="A4")
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()
    pdf.set_left_margin(15)
    pdf.set_right_margin(15)
    def safe(t: str) -> str:
        t = (t.replace("—", "-").replace("–", "-").replace("⚠️", "!")
               .replace("✅", "[ok]").replace("❌", "[missing]").replace("🔍", "[?]"))
        return t.encode("latin-1", errors="ignore").decode("latin-1")
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, safe("Packaging Design Brief - Agency Copy"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    clean = markdown_text.replace("#", "").replace("*", "").replace("`", "").replace("_", "")
    for para in safe(clean).split("\n"):
        if not para.strip():
            pdf.ln(4)
            continue
        pdf.multi_cell(w=0, h=5.5, text=para[:1000], new_x="LMARGIN", new_y="NEXT")
    pdf.output(out_path)
    return out_path
