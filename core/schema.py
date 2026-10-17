"""Product schema — single contract for extract -> gaps -> brief."""
from __future__ import annotations
from dataclasses import dataclass, field

# Mandatory for an agency-usable packaging brief (region-aware: US FDA / India FSSAI)
MANDATORY_FIELDS = [
    "product_name",
    "product_category",
    "pack_size",
    "net_quantity",
    "ingredients",
    "allergens",
    "nutrition",
    "claims",
    "branding_logo",
    "mandatory_content",   # US FDA: net wt + Nutrition Facts + DV statement + maker; India: FSSAI + veg mark + MRP
    "pack_format",         # pouch / carton / bottle ...
    "target_market",
]

# Region-aware + FDA fields. serving_size/servings drive the Esko demo traps.
NICE_FIELDS = ["tagline", "design_notes", "languages",
               "serving_size", "servings_per_pack", "region", "certifications"]

ALL_FIELDS = MANDATORY_FIELDS + NICE_FIELDS

FIELD_LABELS = {
    "product_name": "Product Name",
    "product_category": "Category",
    "pack_size": "Pack Size",
    "net_quantity": "Net Quantity",
    "ingredients": "Ingredients",
    "allergens": "Allergens",
    "nutrition": "Nutrition Facts",
    "claims": "Claims",
    "branding_logo": "Branding / Logo Rule",
    "mandatory_content": "Mandatory Content (FSSAI etc.)",
    "pack_format": "Pack Format",
    "target_market": "Target Market",
    "tagline": "Tagline",
    "design_notes": "Design Notes",
    "languages": "Languages",
    "serving_size": "Serving Size",
    "servings_per_pack": "Servings / Pack",
    "region": "Region (US-FDA / IN-FSSAI)",
    "certifications": "Certifications",
}

PRIORITY = {
    # P0 conflicts handled separately; this is for missing-field ordering
    "allergens": 0,
    "net_quantity": 0,
    "pack_size": 0,
    "mandatory_content": 0,
    "serving_size": 0,
    "servings_per_pack": 1,
    "ingredients": 1,
    "claims": 1,
    "certifications": 1,
    "branding_logo": 1,
    "nutrition": 2,
}


REGION_RULES = {
    "US-FDA": "Net weight + Nutrition Facts (FDA vertical) + %DV statement + maker/packer + allergen (FALCPA) + claims substantiation",
    "IN-FSSAI": "FSSAI logo + Lic No + veg/non-veg mark + MRP + Mfg/Expiry + customer care + Mfd-by address",
}


def region_hint(state: dict) -> str:
    r = (state.get("region").value if state.get("region") else "") or ""
    t = (state.get("target_market").value if state.get("target_market") else "") or ""
    blob = f"{r} {t}".lower()
    if any(k in blob for k in ("united states", "usa", "u.s", "north america", "fda")):
        return "US-FDA"
    if any(k in blob for k in ("india", "fssai", "mrp", "veg mark")):
        return "IN-FSSAI"
    return "US-FDA"  # Esko bundle default


@dataclass
class Source:
    file: str
    loc: str
    quote: str


@dataclass
class ProductField:
    key: str
    value: str | None = None
    status: str = "missing"  # found | missing | conflict | needs_review
    sources: list[Source] = field(default_factory=list)
    confidence: str = "low"  # high | medium | low

    @property
    def label(self) -> str:
        return FIELD_LABELS.get(self.key, self.key)


def empty_state() -> dict[str, ProductField]:
    return {k: ProductField(key=k) for k in ALL_FIELDS}


def completeness(state: dict[str, ProductField]) -> float:
    scored = MANDATORY_FIELDS
    ok = sum(1 for k in scored if state[k].status == "found")
    return round(100 * ok / len(scored), 1)
