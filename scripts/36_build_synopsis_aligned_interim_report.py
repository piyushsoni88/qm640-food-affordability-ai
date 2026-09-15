"""Build the final synopsis-aligned QM640 interim report.

The report is built from the retained Walsh template and the verified aligned
Notebook 01-10 execution summaries. The original research questions and
hypotheses remain controlling.
"""

from __future__ import annotations

import importlib.util
import shutil
from pathlib import Path

from docx.enum.section import WD_SECTION
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = Path(r"C:\Users\piyus\Downloads\QM 640 Interim Report template-1 (4).docx")
OUTPUT = ROOT / "reports" / "Piyush_Soni_QM640_Interim_Report_PhD_Standard_Final.docx"
ASSET_DIR = ROOT / "reports" / "report_assets" / "phd_standard"
EDA_SOURCE_FIGURE = ROOT / "reports" / "notebook_outputs" / "figures" / "03_regional_pressure_2_5_10_years.png"
REPO = "https://github.com/piyushsoni88/qm640-food-affordability-ai"


def load_base():
    path = ROOT / "scripts" / "24_build_final_interim_report.py"
    spec = importlib.util.spec_from_file_location("qm640_report_base", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load report helpers from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.TEMPLATE = TEMPLATE
    module.OUTPUT = OUTPUT
    module.REPO = REPO
    return module


B = load_base()


def font(size: int, bold: bool = False):
    candidates = [
        Path(r"C:\Windows\Fonts\arialbd.ttf" if bold else r"C:\Windows\Fonts\arial.ttf"),
        Path(r"C:\Windows\Fonts\calibrib.ttf" if bold else r"C:\Windows\Fonts\calibri.ttf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size)
    return ImageFont.load_default()


def centered(draw, box, text, fnt, fill="#1F2937"):
    x0, y0, x1, y1 = box
    lines = text.split("\n")
    heights = [draw.textbbox((0, 0), line, font=fnt)[3] for line in lines]
    total = sum(heights) + 7 * (len(lines) - 1)
    y = y0 + (y1 - y0 - total) / 2
    for line, h in zip(lines, heights):
        width = draw.textbbox((0, 0), line, font=fnt)[2]
        draw.text((x0 + (x1 - x0 - width) / 2, y), line, font=fnt, fill=fill)
        y += h + 7


def make_pipeline_figure(path: Path):
    img = Image.new("RGB", (1600, 410), "white")
    d = ImageDraw.Draw(img)
    colors = ["#E8F1FA", "#DCEFE7", "#FFF1D6", "#FCE3E3"]
    labels = [
        "Public signals\nprices | climate | APY\nwages | HCES",
        "Quality-controlled panel\nexact calendar joins\nno price interpolation",
        "Predictive engine\n1-3 month forecasts\nshock probabilities",
        "Decision layer\nexplanations | HFASI\nstakeholder loss",
    ]
    boxes = []
    for i in range(4):
        x0 = 25 + i * 392
        box = (x0, 35, x0 + 330, 375)
        boxes.append(box)
        d.rounded_rectangle(box, radius=20, fill=colors[i], outline="#4B5563", width=3)
        centered(d, box, labels[i], font(22, bold=(i == 2)))
        if i < 3:
            x = x0 + 340
            d.line((x, 200, x + 38, 200), fill="#4B5563", width=5)
            d.polygon([(x + 38, 200), (x + 24, 190), (x + 24, 210)], fill="#4B5563")
    img.save(path, dpi=(180, 180))


def make_data_architecture_figure(path: Path):
    img = Image.new("RGB", (1600, 620), "white")
    d = ImageDraw.Draw(img)
    source_labels = [
        "AGMARKNET\nprices",
        "NASA POWER\nclimate",
        "DES\ncrop APY",
        "Labour Bureau\nrural wages",
        "HCES / macro\nsegment context",
    ]
    for i, label in enumerate(source_labels):
        x0 = 25 + i * 315
        box = (x0, 35, x0 + 260, 170)
        d.rounded_rectangle(box, radius=16, fill="#E8F1FA", outline="#315B7D", width=3)
        centered(d, box, label, font(21, True))
        d.line((x0 + 130, 170, x0 + 130, 220), fill="#64748B", width=4)
        d.polygon([(x0 + 130, 225), (x0 + 120, 210), (x0 + 140, 210)], fill="#64748B")
    gate = (285, 225, 1315, 375)
    d.rounded_rectangle(gate, radius=18, fill="#E5F3EE", outline="#2F6B59", width=4)
    centered(
        d,
        gate,
        "Canonical commodity-region-month panel\nstandardized units | exact calendar joins | lineage | coverage flags | release-safe features",
        font(25, True),
    )
    output_labels = [
        "RQ1\nfixed-effects inference",
        "RQ2\n1-3 month forecasts",
        "Shock layer\n90th-percentile risk",
        "RQ3\nHFASI validation",
        "RQ4\nstakeholder loss",
    ]
    for i, label in enumerate(output_labels):
        x0 = 25 + i * 315
        d.line((x0 + 130, 375, x0 + 130, 420), fill="#64748B", width=4)
        d.polygon([(x0 + 130, 425), (x0 + 120, 410), (x0 + 140, 410)], fill="#64748B")
        box = (x0, 425, x0 + 260, 585)
        fill = "#FFF1D6" if i < 3 else "#FCE3E3"
        d.rounded_rectangle(box, radius=16, fill=fill, outline="#7C5E2A", width=3)
        centered(d, box, label, font(20, True))
    img.save(path, dpi=(180, 180))


def make_rq1_coefficient_figure(path: Path):
    rows = [
        ("Current MoM price", [23.49, 26.04, 21.40]),
        ("Current YoY price", [-4.22, -9.17, -13.20]),
        ("Temperature anomaly", [1.65, 1.92, 1.99]),
        ("Relative humidity", [2.22, 2.37, None]),
        ("Reporting coverage", [None, None, 2.73]),
    ]
    img = Image.new("RGB", (1500, 720), "white")
    d = ImageDraw.Draw(img)
    left, top = 420, 110
    cell_w, cell_h = 300, 95
    for j, h in enumerate(["1 month", "2 months", "3 months"]):
        centered(d, (left + j * cell_w, 35, left + (j + 1) * cell_w, 100), h, font(24, True))
    for i, (label, values) in enumerate(rows):
        y0 = top + i * cell_h
        d.text((35, y0 + 28), label, font=font(23), fill="#1F2937")
        for j, value in enumerate(values):
            x0 = left + j * cell_w
            box = (x0, y0, x0 + cell_w - 8, y0 + cell_h - 8)
            if value is None:
                fill = "#F3F4F6"
                text = "not BH-significant"
                txt_fill = "#6B7280"
            elif value < 0:
                intensity = min(abs(value) / 15, 1)
                fill = (245, int(225 - 65 * intensity), int(225 - 65 * intensity))
                text = f"{value:+.2f}"
                txt_fill = "#7F1D1D"
            else:
                intensity = min(value / 27, 1)
                fill = (int(225 - 75 * intensity), int(235 - 35 * intensity), 248)
                text = f"{value:+.2f}"
                txt_fill = "#163A5F"
            d.rounded_rectangle(box, radius=10, fill=fill, outline="#CBD5E1", width=2)
            centered(d, box, text, font(22, True), fill=txt_fill)
    d.text((420, 640), "Blue = positive standardized association", font=font(19, True), fill="#315B7D")
    d.text((900, 640), "Red = negative standardized association", font=font(19, True), fill="#9B2C2C")
    img.save(path, dpi=(180, 180))


def make_hfasi_validity_figure(path: Path):
    img = Image.new("RGB", (1500, 620), "white")
    d = ImageDraw.Draw(img)
    d.text((45, 28), "Construct validity", font=font(29, True), fill="#111827")
    x0, x1, y = 160, 1050, 220
    d.line((x0, y, x1, y), fill="#64748B", width=4)
    for tick in [-1.0, -0.8, -0.6, -0.4, -0.2, 0.0]:
        x = x0 + (tick + 1) * (x1 - x0)
        d.line((x, y - 12, x, y + 12), fill="#64748B", width=2)
        label = f"{tick:.1f}"
        tw = d.textbbox((0, 0), label, font=font(18))[2]
        d.text((x - tw / 2, y + 25), label, font=font(18), fill="#374151")
    ci_lo, rho, ci_hi = -0.9043, -0.8841, -0.8602
    map_x = lambda v: x0 + (v + 1) * (x1 - x0)
    d.line((map_x(ci_lo), y, map_x(ci_hi), y), fill="#2F6B59", width=14)
    d.ellipse((map_x(rho) - 18, y - 18, map_x(rho) + 18, y + 18), fill="#0F766E")
    d.text((155, 110), "Spearman rho = -0.884", font=font(26, True), fill="#0F766E")
    d.text((155, 150), "95% bootstrap CI [-0.904, -0.860]; one-sided p < .001",
           font=font(21), fill="#374151")
    card = (1090, 80, 1450, 320)
    d.rounded_rectangle(card, radius=18, fill="#E8F1FA", outline="#315B7D", width=3)
    centered(d, card, "Monthly rank stability\nmean = 1.000\nminimum = 1.000", font(24, True))
    lower = (155, 370, 1450, 570)
    d.rounded_rectangle(lower, radius=16, fill="#FFF5E6", outline="#9A6A23", width=3)
    centered(
        d,
        lower,
        "Highest aligned HFASI = 103.494\nDownside purchasing-power scenario | Chandigarh (U.T.) | Rural | Decile 1 | May 2026",
        font(24, True),
    )
    img.save(path, dpi=(180, 180))


def draw_bar_chart(path: Path, title: str, labels: list[str], values: list[float],
                   colors: list[str], value_fmt="{:.2f}", zero_line: bool = False):
    img = Image.new("RGB", (1500, 640), "white")
    d = ImageDraw.Draw(img)
    d.text((55, 25), title, font=font(34, True), fill="#111827")
    left, right, top, bottom = 420, 1420, 100, 585
    d.line((left, top, left, bottom), fill="#9CA3AF", width=2)
    max_abs = max(max(abs(v) for v in values), 1)
    if zero_line:
        min_v, max_v = min(min(values), 0), max(max(values), 0)
        span = max_v - min_v or 1
        zero_x = left + (0 - min_v) / span * (right - left)
        d.line((zero_x, top, zero_x, bottom), fill="#374151", width=3)
    else:
        min_v, max_v, span = 0, max(values) * 1.15, max(values) * 1.15 or 1
        zero_x = left
    row_h = (bottom - top) / len(labels)
    for i, (label, value, color) in enumerate(zip(labels, values, colors)):
        y0 = top + i * row_h + 18
        y1 = top + (i + 1) * row_h - 18
        d.text((45, y0 + 8), label, font=font(25), fill="#1F2937")
        x = left + (value - min_v) / span * (right - left)
        d.rectangle((min(zero_x, x), y0, max(zero_x, x), y1), fill=color)
        text = value_fmt.format(value)
        value_font = font(23, True)
        text_width = d.textbbox((0, 0), text, font=value_font)[2]
        text_fill = "#111827"
        if value >= 0:
            tx = x + 12
            if tx + text_width > 1470:
                tx = x - text_width - 12
                text_fill = "white"
        else:
            tx = x - text_width - 12
        d.text((tx, y0 + 8), text, font=value_font, fill=text_fill)
    img.save(path, dpi=(180, 180))


def make_assets():
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    make_pipeline_figure(ASSET_DIR / "01_conceptual_framework.png")
    make_data_architecture_figure(ASSET_DIR / "02_data_architecture.png")
    if EDA_SOURCE_FIGURE.exists():
        shutil.copyfile(EDA_SOURCE_FIGURE, ASSET_DIR / "03_eda_regional_pressure.png")
    make_rq1_coefficient_figure(ASSET_DIR / "04_rq1_coefficients.png")
    draw_bar_chart(
        ASSET_DIR / "05_shock_performance.png",
        "Commodity shock classification: chronological holdout balanced accuracy",
        ["1 month", "2 months", "3 months"],
        [0.7136, 0.7601, 0.7115],
        ["#4C78A8", "#59A14F", "#F28E2B"],
        value_fmt="{:.3f}",
    )
    make_hfasi_validity_figure(ASSET_DIR / "06_hfasi_validity.png")
    draw_bar_chart(
        ASSET_DIR / "07_decision_loss.png",
        "Retrospective decision-loss reduction by stakeholder",
        ["Household budgeting", "Retail inventory", "Enterprise procurement", "Policy monitoring"],
        [48.38, 45.71, -112.37, -174.21],
        ["#59A14F", "#59A14F", "#E15759", "#E15759"],
        value_fmt="{:+.1f}%",
        zero_line=True,
    )


def title_page(doc):
    for _ in range(2):
        doc.add_paragraph()
    items = [
        ("Data Analytics Capstone", True, 14),
        ("Forecasting Essential Food Price Shocks and Household Affordability Stress in India", True, 14),
        ("An Explainable AI Decision-Intelligence Framework Using Public Market, Agricultural, Climatic, and Economic Data", False, 12),
        ("Interim Report", True, 14),
        ("Piyush Soni", False, 12),
        ("Walsh College", False, 12),
        ("QM640: Data Analytics Capstone", False, 12),
        ("Mentor: Prof. Rishabh Pandey", False, 12),
        ("Summer 2026 Term", False, 12),
        ("July 30, 2026", False, 12),
    ]
    for idx, (text, bold, size) in enumerate(items):
        p = doc.add_paragraph()
        r = p.add_run(text)
        B.set_run_font(r, size=size, bold=bold)
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.first_line_indent = None
        p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.DOUBLE
        p.paragraph_format.space_before = Pt(8 if idx in {3, 4} else 0)
        p.paragraph_format.space_after = Pt(0)
    sec = doc.add_section(WD_SECTION.NEW_PAGE)
    sec.page_width = Inches(8.5)
    sec.page_height = Inches(11)
    sec.top_margin = Inches(1.5)
    sec.bottom_margin = Inches(1.5)
    sec.left_margin = Inches(1)
    sec.right_margin = Inches(1)
    sec.header_distance = Inches(0.55)
    sec.footer_distance = Inches(0.55)


def repo_link(doc, label: str, path: str, explanation: str):
    B.add_repo_link(doc, label, path, explanation)


def compact_reference(doc, text: str):
    p = doc.add_paragraph()
    r = p.add_run(text)
    B.set_run_font(r, size=8.2)
    pf = p.paragraph_format
    pf.alignment = WD_ALIGN_PARAGRAPH.LEFT
    pf.left_indent = Inches(0.42)
    pf.first_line_indent = Inches(-0.42)
    pf.line_spacing = 1.0
    pf.space_before = Pt(0)
    pf.space_after = Pt(3)
    return p


def add_status_tag(doc, text: str):
    p = doc.add_paragraph()
    r = p.add_run(text)
    B.set_run_font(r, size=10, bold=True)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.first_line_indent = None
    p.paragraph_format.line_spacing = 1.0
    p.paragraph_format.space_after = Pt(4)


def add_picture_with_alt(doc, path: Path, alt_text: str, width: float):
    paragraph = B.add_picture(doc, path, width=width)
    shape = doc.inline_shapes[-1]
    shape._inline.docPr.set("descr", alt_text)
    shape._inline.docPr.set("title", alt_text)
    return paragraph


def build():
    make_assets()
    doc = B.prepare_document()
    title_page(doc)
    B.configure_headers(doc)

    # Main report page 1: repository link and evaluator access.
    B.add_heading(doc, "GitHub Repository and Project Status", 1)
    B.add_heading(doc, "GitHub Data Availability Statement", 2)
    p = doc.add_paragraph()
    p.add_run("Public repository: ")
    B.add_hyperlink(p, REPO, REPO)
    B.format_paragraph(p, first_indent=False)
    repo_link(doc, "Curated analytical data", "tree/main/data/curated",
              "compressed price, climate, macroeconomic, and national/state panels")
    repo_link(doc, "Confirmatory supplements", "tree/main/data/external_required",
              "DES crop APY, Labour Bureau wages, and HCES 2023-24 segment aggregates")
    repo_link(doc, "Data dictionary", "blob/main/data/data_dictionary.csv",
              "field definitions, analytical roles, transformations, and handling rules")
    repo_link(doc, "Aligned notebooks", "tree/main/notebooks",
              "ten Colab-compatible notebooks from acquisition through final consolidation")
    repo_link(doc, "Notebook evidence", "tree/main/reports/notebook_outputs",
              "execution summaries, model metrics, tables, figures, and checks")
    B.add_table_caption(doc, 1, "Verified analytical coverage")
    B.add_table(
        doc,
        ["Layer", "Verified quantity", "Coverage / qualification"],
        [
            ["Curated analytical files", "1,494,745 rows", "14 files audited"],
            ["AGMARKNET daily aggregate", "969,377 rows", "Represents 18,836,462 official records"],
            ["State-month-commodity panel", "41,792 rows", "32 states/UTs; 8 commodities; 2001-Jul 2026"],
            ["National monthly panel", "319 rows", "2000-Jul 2026; CPI observed through Dec 2025"],
        ],
        [2.0, 1.6, 2.9],
        font_size=8.4,
    )
    B.add_note(doc, "Permitted compressed data are public in the repository. Bulk/restricted raw archives are represented by official retrieval scripts, manifests, and checksums; credentials are excluded. Appendix A maps the full repository and notebook evidence chain.")
    B.add_page_break(doc)

    # Main report page 2: introduction and background, following the template.
    B.add_heading(doc, "Introduction", 1)
    B.add_heading(doc, "Background and Context", 2)
    B.add_body(doc, "Food-price volatility is simultaneously a household-welfare problem, a business-continuity risk, and a public-policy challenge. Essential commodities such as rice, wheat, onion, potato, tomato, pulses, edible oil, and sugar have different crop calendars, storability, transport exposure, and market structures. Their prices can therefore react unevenly to monsoon deviations, crop losses, market congestion, fuel costs, trade restrictions, festival demand, and reporting disruptions. A late signal is costly: households lose purchasing power, retailers face stock-outs or excess inventory, enterprises make emergency purchases, and public agencies intervene after rather than before acute pressure.")
    B.add_body(doc, "The distributional consequences are equally important. Food absorbs a larger share of expenditure among rural and lower-expenditure households; consequently, an identical commodity shock can create very different welfare pressure across sectors and expenditure deciles. The HCES 2023-24 aggregates used in this study preserve that heterogeneity across 36 states/UTs, rural and urban sectors, and ten expenditure groups. The analytical requirement is therefore not merely an accurate price forecast. The system must also explain predictive signals, quantify abnormal upward risk, translate expected food-cost growth into an interpretable affordability construct, and evaluate whether warnings improve decisions.")
    B.add_body(doc, "India offers an unusually rich but fragmented public-data environment. AGMARKNET supplies official mandi observations; NASA POWER supplies reproducible climate histories; the Directorate of Economics and Statistics provides crop area, production, and yield; the Labour Bureau provides rural wages; MOSPI and HCES provide inflation and household-expenditure context; and the World Bank provides external food and energy benchmarks. These sources differ in frequency, spatial unit, measurement convention, coverage, and release timing. Integrating them requires explicit lineage, canonical keys, exact calendar joins, leakage controls, and transparent treatment of missingness.")
    B.add_body(doc, "This interim report continues the approved synopsis without changing its title, four research questions, four hypothesis pairs, one- to three-month horizons, unit of analysis, or stakeholder groups. Its contribution is an auditable evidence chain rather than a single black-box model: statistical driver analysis, chronological forecast comparison, commodity-specific shock classification, model explanation, a validated Household Food Affordability Stress Index (HFASI), and stakeholder-specific decision-loss testing are linked within one reproducible framework.")
    B.add_page_break(doc)

    # Main report page 3: problem, purpose, and progress.
    B.add_heading(doc, "Problem Statement, Purpose, and Interim Status", 1)
    B.add_heading(doc, "Problem Statement", 2)
    B.add_body(doc, "Indian households and food-sector organizations face recurring financial and operational uncertainty because essential-food prices are volatile, geographically uneven, and driven by interacting climatic, agricultural, market, seasonal, and macroeconomic forces. Relevant public data are available, but they are commonly analyzed in separate systems and used for retrospective monitoring. The resulting gap is an integrated and independently evaluated early-warning workflow that can predict price change, identify abnormal upward risk, disclose uncertainty, explain model use of the available signals, and connect the forecast to affordability and decision outcomes.")
    B.add_heading(doc, "Purpose of the Study", 2)
    B.add_body(doc, "The purpose of this quantitative longitudinal study is to develop and validate a reproducible decision-intelligence framework for essential-food price shocks and household affordability stress in India. Four connected tasks are retained from the synopsis: identify statistically relevant drivers; compare machine-learning and conventional forecasting performance; construct and validate HFASI across household segments; and test whether model-driven warnings reduce retrospective loss for household budgeting, retail inventory, enterprise procurement, and policy monitoring. All outputs are predictive or associational unless a separate causal identification design is available.")
    B.add_heading(doc, "Interim Project Status (Progress Snapshot)", 2)
    B.add_table(
        doc,
        ["Status", "Completed or planned work"],
        [
            ["Completed", "Acquisition, lineage audit, cleaning, exact targets, EDA, RQ1 inference, RQ2 forecasting, shock classification, explainability, HFASI, stakeholder-loss analysis, and consolidated evidence across ten notebooks."],
            ["Integrated", "DES state crop area/production/yield; HCES 2023-24 aggregates; Labour Bureau rural wages with partial panel coverage."],
            ["Open gap", "Genuine market-arrival quantity. Reporting-row counts remain quality/coverage indicators and are never relabelled as arrivals."],
            ["Final-stage work", "Broader release-date-safe arrivals, wage coverage, geographic robustness, probability calibration, stakeholder cost tuning, and monitored deployment."],
        ],
        [1.2, 5.3],
        font_size=8.4,
    )
    B.add_body(doc, "Notebook 10 verified the nine upstream execution summaries, mapped every result to its controlling research question and hypothesis, registered nine material limitations and six next-stage tasks, and inventoried 155 analytical artifacts. The project is therefore analytically complete at the interim stage, while the remaining work is explicitly confirmatory: genuine arrivals, broader wage coverage, longer chronological validation, calibrated probabilities, and stakeholder-specific operating thresholds.")
    B.add_page_break(doc)

    # Main report page 4: scope, objectives, and exact research questions.
    B.add_heading(doc, "Scope, Objectives, and Research Questions", 1)
    B.add_body(doc, "The primary unit is the commodity-region-month. The confirmatory outcomes are one-, two-, and three-month future price changes and commodity-specific abnormal upward shocks. The current panel includes 32 reporting states/UTs and eight essential commodities from 2001 through July 2026; 2026 is explicitly year-to-date. A national monthly benchmark and 24-month conditional CPI path supplement, but do not replace, the original commodity-region design.")
    B.add_body(doc, "The original sample-planning threshold was 1,044 effective observations after inflation for clustering, unusable periods, and source attrition. The aligned inferential models contain approximately 3,800 complete state-commodity-month cases across 25 states/UTs, while forecast evaluation comprises 12,540 predictions. These record counts are not treated as independent sample sizes: repeated observations, temporal dependence, and common shocks are handled through fixed effects, cluster-robust uncertainty, chronological origins, and multiplicity control.")
    B.add_heading(doc, "Research Questions", 2)
    B.add_bullet(doc, "Research Question 1 (RQ1). Which lagged-price, market-arrival, climatic, seasonal, agricultural-production, and macroeconomic variables significantly influence one- to three-month essential-food price changes in India?")
    B.add_bullet(doc, "Research Question 2 (RQ2). Do machine-learning and ensemble models predict essential-food prices and price shocks more accurately than conventional statistical forecasting models?")
    B.add_bullet(doc, "Research Question 3 (RQ3). How can forecasted commodity-price changes be translated into a statistically reliable, interpretable, and stable Household Food Affordability Stress Index across rural, urban, and expenditure segments?")
    B.add_bullet(doc, "Research Question 4 (RQ4). How effectively can the proposed early-warning framework support household budgeting, retail inventory, enterprise procurement, and policy-monitoring decisions under alternative supply and price scenarios?")
    B.add_heading(doc, "Measurable Objectives", 2)
    B.add_bullet(doc, "Estimate driver direction, magnitude, uncertainty, and horizon stability.")
    B.add_bullet(doc, "Compare ML with transparent statistical baselines using chronological out-of-sample loss.")
    B.add_bullet(doc, "Validate HFASI construct ordering and segment-rank stability.")
    B.add_bullet(doc, "Test whether model warnings reduce stakeholder-specific retrospective loss.")
    B.add_page_break(doc)

    # Main report page 5: hypotheses and conceptual framework.
    B.add_heading(doc, "Hypotheses and Conceptual Framework", 1)
    B.add_table_caption(doc, 2, "Original hypothesis pairs and current decision status")
    B.add_table(
        doc,
        ["RQ", "Null and alternative logic retained from the synopsis", "Aligned interim decision"],
        [
            ["RQ1", "H0-1: joint pre-specified driver coefficients equal zero. H1-1: at least one coefficient is nonzero after controls.", "Reject available-variable H0-1 at 1-3 months; full H0-1 partially unresolved because arrivals are missing."],
            ["RQ2", "H0-2: best ML/ensemble expected loss is >= best statistical baseline. H1-2: at least one ML/ensemble has lower loss.", "Fail to reject H0-2 at all horizons; Holm-adjusted p=1.000."],
            ["RQ3", "H0-3: HFASI lacks construct association and stable segment ranks. H1-3: association and ranks are valid and stable.", "Provisionally reject H0-3; rho=-.884 and monthly rank stability=1.000."],
            ["RQ4", "H0-4: warnings do not reduce decision loss. H1-4: paired retrospective loss is reduced.", "Partial support: household and retail improve; procurement and policy do not."],
        ],
        [0.55, 3.45, 2.5],
        font_size=8.2,
    )
    B.add_figure_caption(doc, 1, "Synopsis-aligned analytical and decision framework")
    add_picture_with_alt(
        doc,
        ASSET_DIR / "01_conceptual_framework.png",
        "Flow diagram from public price, climate, agricultural, wage, and HCES signals through quality control, predictive models, explanations, HFASI, and stakeholder decisions.",
        6.15,
    )
    B.add_note(doc, "Observable public signals flow through quality controls into predictive outputs, explanations, HFASI, and stakeholder-specific decisions. Predictive explanations are not causal effects.")
    B.add_body(doc, "The hypothesis architecture separates statistical significance, predictive accuracy, construct validity, and practical decision value. A statistically significant driver need not improve a forecast; an accurate forecast need not produce a useful alert; and an alert can be beneficial for one stakeholder but costly for another. Exact p values, confidence intervals, out-of-sample losses, rare-event metrics, and loss reduction are therefore interpreted jointly rather than collapsed into a single claim of model success.")
    B.add_page_break(doc)

    # Main report page 6: literature survey.
    B.add_heading(doc, "Literature Survey", 1)
    B.add_body(doc, "Sources were selected using agricultural-price forecasting, food inflation, price shock, explainable AI, household affordability, and chronological validation. Inclusion required a traceable scholarly or official source, a described method, and direct relevance to at least one research question. The synthesis supports strong baselines, time-ordered evaluation, explicit rare-event metrics, transparent explanation, and separate welfare validation.")
    B.add_note(doc, "Appendix E (Table 13) provides the ten-source literature relevance matrix. The APA-style bibliography also includes official MOSPI, OGD India, NASA POWER, Labour Bureau, and World Bank data sources.")
    B.add_heading(doc, "Thematic Synthesis and Research Gap", 2)
    B.add_body(doc, "First, the forecasting literature consistently warns against assuming that algorithmic complexity guarantees superior accuracy. Hyndman and Koehler (2006) establish the need for scale-aware error measures, while Makridakis et al. (2018) show that transparent statistical methods can remain difficult to beat. This evidence directly motivates zero-change and seasonal-change baselines, expanding-window evaluation, and formal paired loss comparisons in RQ2.")
    B.add_body(doc, "Second, India-focused studies demonstrate the value of market-specific learning, anomaly detection, and interpretable prediction (Ma et al., 2019; Madaan et al., 2019), but the broader literature reports persistent weaknesses in exogenous-variable coverage, geographic generalization, leakage control, uncertainty communication, and common temporal benchmarks (Theofilou et al., 2025). The present study addresses these limitations by using release-safe features, exact forecast horizons, source-quality controls, horizon-specific models, and an explicit distinction between reporting coverage and genuine supply.")
    B.add_body(doc, "Third, welfare and diet-affordability research indicates that price shocks are mediated by household expenditure structure and purchasing power (Akter & Basher, 2014; Cattaneo et al., 2023). This motivates HFASI as a formative segment-level construct rather than a generic inflation score. The unresolved gap is a single auditable framework in which forecast accuracy, shock detection, household construct validity, and stakeholder decision value are tested separately but reported together.")
    B.add_page_break(doc)

    # Main report page 7: data description and access.
    B.add_heading(doc, "Data Description, Sources, and Access", 1)
    B.add_heading(doc, "Dataset Overview", 2)
    B.add_body(doc, "Notebook 01 audited 14 curated files and 1,494,745 analytical rows. The 969,377-row AGMARKNET daily state aggregate represents 18,836,462 underlying official observations. Monthly aggregation yields 41,792 state-commodity records. The model panel is unbalanced because official market reporting varies by state, commodity, and month; coverage indicators are retained to distinguish information availability from supply.")
    B.add_body(doc, "The state panel spans January 2001 through July 2026 and covers 32 reporting states/UTs and eight commodities. The national monthly panel spans January 2000 through July 2026, with food CPI observed through December 2025. The target variables are exact one-, two-, and three-month future price changes and training-only commodity-specific shock labels. The unit of analysis, data horizon, and target definitions therefore remain identical to the synopsis.")
    B.add_heading(doc, "Data Source(s) and Access", 2)
    B.add_body(doc, "AGMARKNET/OGD supplies the price backbone; NASA POWER supplies rainfall, temperature, and humidity; DES and Labour Bureau provide state-crop APY and partially matched rural wages; HCES 2023-24 supplies disclosure-safe MPCE and food-share segments; and MOSPI, FAOSTAT, and World Bank series provide national and external controls. The public source manifest, checksums, curated data, and field definitions are linked on page 2 and reproduced in Appendices A-B.")
    B.add_figure_caption(doc, 2, "Source integration, analytical panels, and research-question outputs")
    add_picture_with_alt(
        doc,
        ASSET_DIR / "02_data_architecture.png",
        "Architecture diagram linking AGMARKNET, NASA POWER, crop APY, rural wages, HCES, and macroeconomic data through a canonical quality-controlled panel to the four research-question outputs.",
        6.15,
    )
    B.add_note(doc, "Genuine arrivals remain missing. The variable source_rows is a reporting-coverage measure, not market-arrival quantity. Climate values are point-based representatives, not area-weighted state means.")
    B.add_page_break(doc)

    # Main report page 8: analysis, cleaning, and mandatory dictionary.
    B.add_heading(doc, "Analysis", 1)
    B.add_heading(doc, "Data Cleaning", 2)
    B.add_table_caption(doc, 3, "Cleaning and integration log")
    B.add_table(
        doc,
        ["Issue", "Detection / treatment", "Verified result"],
        [
            ["Validity and duplicates", "Key/range rules; exact region-commodity-month keys", "0 rows removed; 41,792 retained"],
            ["Extreme prices", "Robust flags; preserve plausible shocks", "15 rows flagged (0.0359%); raw values preserved"],
            ["Missing prices", "No synthetic price interpolation", "Observed values only"],
            ["Calendar lags/targets", "Exact keyed date joins", "1m 40,189; 2m 39,587; 3m 39,162 targets"],
            ["Confirmatory sources", "Canonical state/commodity/date harmonization", "APY integrated; wages partial; HCES aggregates available"],
        ],
        [1.35, 3.1, 2.05],
        font_size=8.0,
    )
    B.add_body(doc, "No row failed the pre-specified validity rules. This is not evidence that the source is error-free; rather, it means that no observation violated the declared key, date, unit, or plausibility constraints strongly enough to justify removal. Fifteen extreme-price rows were retained with flags because they could represent genuine market shocks. Missing prices were never interpolated, preventing synthetic smoothing of the dependent variable. All lagged and future variables were created by exact keyed month joins, so an absent calendar month remains absent rather than being treated as an adjacent observation.")
    B.add_heading(doc, "Data Dictionary (Mandatory)", 2)
    B.add_body(doc, "The mandatory expanded dictionary is provided in Appendix B (Table 9). It defines panel keys; observed and future price-change variables; climate anomalies; production, yield, rural wage, and reporting-coverage fields; HCES food share, MPCE, sector and decile; shock targets; and HFASI. In particular, source_rows is documented only as a reporting-quality control and never as market-arrival quantity.")
    B.add_heading(doc, "GitHub Data Availability Statement", 2)
    B.add_body(doc, "Evaluator-accessible compressed data, the source manifest, data dictionary, aligned notebooks, execution summaries, and report artifacts are available at the repository paths listed on page 2. Appendix A provides the complete folder tree and notebook-to-evidence map; Appendix B provides the expanded field dictionary.")
    B.add_page_break(doc)

    # Main report page 9: EDA.
    B.add_heading(doc, "Exploratory Data Analysis", 1)
    B.add_body(doc, "EDA covered all 41,792 state rows and 319 national months. Onion has the largest year-over-year price-change standard deviation (117.75), confirming that pooled averages conceal commodity-specific risk. West Bengal has the highest median price pressure in each recent window: 3.13% over two years, 6.70% over five years, and 4.06% over ten years. The window comparison prevents a short recent period from being mistaken for a stable long-run ranking.")
    B.add_figure_caption(doc, 3, "Regional mandi-price pressure across two-, five-, and ten-year windows")
    add_picture_with_alt(
        doc,
        ASSET_DIR / "03_eda_regional_pressure.png",
        "Three-panel horizontal bar chart comparing median year-over-year mandi-price change across states for two-, five-, and ten-year windows; West Bengal ranks highest in each window.",
        6.2,
    )
    B.add_table_caption(doc, 4, "EDA insight summary")
    B.add_table(
        doc,
        ["Evidence", "Result", "Interpretation / decision"],
        [
            ["Commodity volatility", "Onion; YoY SD 117.75", "Retain commodity effects and shock-specific metrics."],
            ["Regional pressure", "West Bengal leads 2y/5y/10y: 3.13%, 6.70%, 4.06%", "Pressure is not an artifact of one window."],
            ["Exploratory inflation threshold", "8.027%; 75 high-inflation months", "Descriptive only; classifier uses training-only commodity thresholds."],
            ["Climate-price correlations", "Rain -.0025; temperature -.0040", "Near-zero unconditional correlation does not preclude conditional/nonlinear effects."],
            ["Latest food CPI", "158.388 in Dec 2025", "Observed endpoint for national forecasting."],
        ],
        [1.75, 1.65, 3.1],
        font_size=7.8,
    )
    B.add_body(doc, "Figure 3 shows that recent regional pressure is heterogeneous: the two-year window includes both positive and negative state medians, while the five- and ten-year windows are more broadly positive. The stability of West Bengal's leading rank supports targeted monitoring, but it does not imply that state identity causes price inflation. Differences may also reflect commodity mix, market coverage, crop cycles, and reporting intensity.")
    B.add_body(doc, "All EDA associations are descriptive, not causal. The unbalanced reporting panel, point-based climate coverage, and partial 2026 observations are retained as visible qualifications. Detailed commodity indices, seasonality, inflation-shock months, climate-price diagnostics, and coverage plots are available in the Notebook 03 evidence folder.")
    B.add_page_break(doc)

    # Main report page 10: modelling, feature engineering, and metrics.
    B.add_heading(doc, "Modelling", 1)
    B.add_heading(doc, "Choice of Models With Justification", 2)
    B.add_body(doc, "Model choice follows the research question rather than a single-algorithm strategy. RQ1 requires interpretable conditional associations and joint hypothesis tests, so fixed-effects panel regression with cluster-robust uncertainty is preferred. RQ2 requires an honest prediction contest, so zero-change and seasonal-change baselines are compared with Ridge and HistGradientBoosting under expanding chronological origins. HistGradientBoosting accommodates nonlinearities and interactions without requiring the data volume of a deep neural network. A damped-trend exponential-smoothing model is retained only as a supplementary national planning path. RQ3 and RQ4 use explicitly constructed indices and decision-loss functions because those questions concern validity and practical utility, not only prediction error.")
    B.add_table_caption(doc, 5, "Model family, feature set, validation, and analytical use")
    B.add_table(
        doc,
        ["Model / layer", "Principal features", "Validation / inference", "Use"],
        [
            ["Fixed-effects panel", "Price change, climate, APY, coverage, season, region, commodity", "Cluster-robust; BH FDR", "RQ1 driver evidence"],
            ["Ridge / zero / seasonal", "Release-safe lagged and contextual predictors", "Expanding chronological origins", "RQ2 transparent baselines"],
            ["HistGradientBoosting", "Same origin-available panel features", "Horizon-specific rolling evaluation", "Price and shock prediction"],
            ["Damped ETS", "National food-CPI history", "24-month planning horizon", "Supplementary scenario path"],
            ["HFASI / loss rules", "Forecast growth, food share, purchasing power, stakeholder costs", "Bootstrap validity; retrospective paired loss", "RQ3/RQ4"],
        ],
        [1.45, 2.3, 1.75, 1.0],
        font_size=7.5,
    )
    B.add_heading(doc, "Features Included and Feature Engineering", 2)
    B.add_body(doc, "The aligned feature set contains current monthly and annual price change, rainfall anomaly, temperature anomaly, relative humidity, log production, log yield, reporting coverage, calendar seasonality, commodity, region, and partial rural-wage features. Log transforms reduce scale asymmetry in production, yield, and coverage; one-hot or native categorical handling preserves commodity and region heterogeneity; and exact forward targets enforce the declared one-, two-, and three-month horizons. Every predictor is defined at the feature month so no future outcome or revised value leaks into training.")
    B.add_heading(doc, "Evaluation Metrics (Include Formulae and Calculations)", 2)
    B.add_body(doc, "Forecast error is evaluated with RMSE; rare-shock discrimination with balanced accuracy, recall, and average precision; probability accuracy with the Brier score; HFASI construct ordering with Spearman rho and bootstrap confidence intervals; and practical utility with percentage loss reduction. Appendix C (Table 10) gives each formula, calculation, and aligned analytical use.")
    B.add_page_break(doc)

    # Main report page 11: RQ1.
    B.add_heading(doc, "Preliminary Results: RQ1 Statistical Driver Analysis", 1)
    B.add_body(doc, "Fixed-effects panel models estimate future price change at one, two, and three months using eight available predictor groups and commodity, region, season, and time controls. Model samples contain 3,798, 3,783, and 3,777 observations across 25 states/UTs. R-squared is .247, .249, and .266, while RMSE increases from 16.35 to 37.48 percentage points as the horizon lengthens.")
    B.add_figure_caption(doc, 4, "Direction and magnitude of BH-significant available-variable associations")
    add_picture_with_alt(
        doc,
        ASSET_DIR / "04_rq1_coefficients.png",
        "Heatmap of significant standardized panel coefficients at one, two, and three months, showing positive current monthly price and temperature associations and negative current annual price associations.",
        5.9,
    )
    B.add_body(doc, "The positive current MoM term and negative YoY term indicate short-run momentum combined with longer-run mean-reversion or base effects. Temperature remains significant across horizons, while humidity is significant at one and two months. Reporting coverage at three months is a measurement result and must not be interpreted as supply. Three partial wage models were estimated, but low integration coverage limits their confirmatory weight.")
    B.add_body(doc, "Model fit is moderate and horizon-dependent: R-squared rises from .247 at one month to .266 at three months, while residual RMSE increases from 16.35 to 37.48 percentage points. The increasing error is substantively important because it shows that statistically detectable associations do not eliminate long-horizon uncertainty. The joint available-variable null is rejected at all horizons, but effect stability and practical magnitude remain more informative than isolated p values.")
    add_status_tag(doc, "RQ1 decision: Reject the available-variable H0-1 at all horizons; do not claim the full original H0-1 is settled.")
    B.add_note(doc, "Coefficients are conditional associations, not causal effects. Genuine market-arrival quantity is required before the full pre-specified driver hypothesis can be adjudicated.")
    B.add_page_break(doc)

    # Main report page 11: RQ2 forecasting and shock classification.
    B.add_heading(doc, "Preliminary Results: RQ2 Forecasting and Shock Classification", 1)
    B.add_heading(doc, "Choice of Models With Justification", 2)
    B.add_body(doc, "Eight expanding chronological origins from May 2024 through February 2026 compare HistGradientBoosting, Ridge, zero-change, and seasonal-change forecasts. HistGradientBoosting is selected at all three horizons across 12,540 evaluated panel forecasts. However, paired loss differences do not establish statistically meaningful superiority over the best baseline: Holm-adjusted p=1.000 for every horizon. H0-2 is therefore not rejected. A damped ETS model supplies a supplementary 24-month national CPI planning path for January 2026 through December 2027.")
    B.add_table_caption(doc, 6, "Selected commodity-shock classifier performance")
    B.add_table(
        doc,
        ["h", "BA", "TPR", "Avg. precision", "BS", "Shocks"],
        [
            ["1", ".714", ".630", ".331", ".150", "73"],
            ["2", ".760", ".773", ".311", ".169", "66"],
            ["3", ".712", ".717", ".259", ".183", "60"],
        ],
        [1.0, 1.15, 0.9, 1.4, 0.85, 1.2],
        font_size=8.0,
    )
    B.add_figure_caption(doc, 5, "Chronological shock-classification balanced accuracy")
    add_picture_with_alt(
        doc,
        ASSET_DIR / "05_shock_performance.png",
        "Horizontal bars showing balanced accuracy of 0.714, 0.760, and 0.712 for one-, two-, and three-month commodity shock classifiers.",
        5.1,
    )
    B.add_body(doc, "The highest forward conditional risk is 0.820 for Delhi-Rice at the three-month horizon (target October 2026). This is a model-based risk watch, not an observed future event or automatic policy trigger. The 90th-percentile shock threshold is computed from training data only, preventing leakage.")
    B.add_body(doc, "The forecast and classification results must be interpreted together. HistGradientBoosting is the lowest-loss candidate at each horizon, but the paired loss tests do not distinguish its improvement from chance under the available eight rolling origins; Holm-adjusted p values equal 1.000. H0-2 is therefore not rejected. In contrast, the shock classifiers demonstrate useful ranking and sensitivity, particularly at two months, but their average precision remains only .259-.331. The appropriate conclusion is conditional monitoring value, not general superiority over statistical baselines.")
    add_status_tag(doc, "RQ2 decision: Fail to reject H0-2 at one, two, and three months.")
    B.add_page_break(doc)

    # Main report page 13: explainability and RQ3.
    B.add_heading(doc, "Preliminary Results: RQ3 Household Affordability", 1)
    B.add_heading(doc, "Explainability", 2)
    B.add_body(doc, "Chronological holdout permutation importance identifies rainfall anomaly as the leading one-month forecast feature, reporting coverage (source_rows) at two months, and commodity identity at three months. Shock importance is led by reporting coverage at one and two months and commodity at three months. In the highest-risk Delhi-Rice case, the leading local sensitivity is current MoM price change; a one-standard-deviation central perturbation changes probability by -0.238. These results describe fitted-model usage and measurement sensitivity, not causal attribution.")
    B.add_heading(doc, "HFASI Construction and Validation", 2)
    B.add_body(doc, "HFASI = 100 + HCES food share x (forecast food-cost growth - purchasing-power growth). Notebook 08 uses 719 disclosure-safe HCES 2023-24 aggregate rows representing 261,245 sampled households across 36 states/UTs, rural/urban sectors, and expenditure deciles 1-10. It does not use or expose household microdata.")
    B.add_body(doc, "Construct evidence is strong and directionally coherent: Spearman rho=-.884 (one-sided p<.001), the bootstrap 95% interval is [-.904, -.860], and mean and minimum monthly rank stability equal 1.000. The highest modeled segment value is 103.494 for Chandigarh (U.T.), rural decile 1, in May 2026 under the downside purchasing-power scenario.")
    B.add_figure_caption(doc, 6, "HFASI construct-validity, rank-stability, and peak-segment evidence")
    add_picture_with_alt(
        doc,
        ASSET_DIR / "06_hfasi_validity.png",
        "Forest-style construct validity figure showing Spearman rho minus 0.884 with bootstrap confidence interval minus 0.904 to minus 0.860, monthly rank stability of 1.000, and the highest HFASI segment.",
        5.15,
    )
    B.add_body(doc, "The negative correlation is expected because higher MPCE should reduce the fraction of the budget devoted to food. Its magnitude and narrow bootstrap interval provide strong convergent construct evidence. Perfect monthly rank stability shows that the segment ordering is not an artifact of a single forecast month. Nevertheless, this is a formative aggregate index: it validates the ordering of disclosure-safe segments, not the welfare condition of individual households.")
    add_status_tag(doc, "RQ3 decision: Provisionally reject H0-3; HFASI is construct-valid and rank-stable within the aggregate segment design.")
    B.add_note(doc, "HFASI is a scenario-based segment construct, not a household welfare threshold or causal effect.")
    B.add_page_break(doc)

    # Main report page 14: RQ4.
    B.add_heading(doc, "Preliminary Results: RQ4 Decision Value and Scenarios", 1)
    B.add_body(doc, "Notebook 09 evaluates 3,135 retrospective observations for each of four stakeholders. Model warnings reduce loss by 48.38% for household budgeting and 45.71% for retail inventory, so H0-4 is rejected for these uses. They increase loss for enterprise procurement (-112.37%) and policy monitoring (-174.21%); H0-4 is not rejected for those uses. The overall conclusion is partial support, not universal decision superiority.")
    B.add_table_caption(doc, 7, "Stakeholder decision-loss results")
    B.add_table(
        doc,
        ["Stakeholder", "Loss reduction", "Hypothesis decision", "Operational interpretation"],
        [
            ["Household budgeting", "+48.38%", "Reject H0-4", "Warnings improve budget-timing decisions."],
            ["Retail inventory", "+45.71%", "Reject H0-4", "Warnings improve replenishment decisions."],
            ["Enterprise procurement", "-112.37%", "Fail to reject H0-4", "Current threshold/cost rule is unsuitable."],
            ["Policy monitoring", "-174.21%", "Fail to reject H0-4", "Current warning rule over-penalizes policy decisions."],
        ],
        [1.55, 1.05, 1.45, 2.45],
        font_size=7.8,
    )
    B.add_figure_caption(doc, 7, "Retrospective decision-loss reduction by stakeholder")
    add_picture_with_alt(
        doc,
        ASSET_DIR / "07_decision_loss.png",
        "Horizontal diverging bars showing positive loss reduction for household budgeting and retail inventory and negative reduction for enterprise procurement and policy monitoring.",
        5.1,
    )
    B.add_body(doc, "Controlled favorable, baseline, moderate, and severe scenarios test the joint HFASI-risk decision space. The severe stress overlay reaches HFASI 105.494 and a clipped modeled shock probability of 1.000. These are deterministic stress tests, not forecasts; the clipped probability signals extrapolation and reinforces the need for calibration and human review.")
    B.add_body(doc, "The opposite signs across stakeholders demonstrate why accuracy alone is an incomplete criterion. Household and retail decisions benefit because their loss functions reward earlier adjustment and tolerate moderate false alarms. Procurement and policy rules incur much larger losses because the current thresholds trigger costly actions under insufficiently calibrated probabilities. The model is therefore not rejected as a general forecasting system; rather, the decision policy is rejected for those two use cases until asymmetric costs, lead times, and action thresholds are re-estimated.")
    add_status_tag(doc, "RQ4 decision: Partial support - warnings reduce loss for two of four stakeholders.")

    # Main report page 14: limitations, next steps, conclusion.
    B.add_heading(doc, "Limitations, Ethics, Next Steps, and Conclusion", 1)
    B.add_heading(doc, "Interim Limitations and Risks", 2)
    B.add_bullet(doc, "Genuine market-arrival quantity is missing; reporting coverage must not substitute for supply.")
    B.add_bullet(doc, "Rural-wage integration is partial, limiting purchasing-power inference in the panel models.")
    B.add_bullet(doc, "The official market panel is unbalanced; coverage changes can resemble price changes.")
    B.add_bullet(doc, "Climate features are representative point locations, not area-weighted state averages.")
    B.add_bullet(doc, "Only eight chronological forecast origins are available in the aligned panel evaluation.")
    B.add_bullet(doc, "HCES inputs are disclosure-safe aggregates; HFASI is not a household-level causal welfare estimate.")
    B.add_bullet(doc, "Explainability, fixed-effects estimates, and stress scenarios are predictive/associational, not causal.")
    B.add_heading(doc, "Ethics, Governance, and Reproducibility", 2)
    B.add_body(doc, "The project uses public or disclosure-safe aggregate data and does not process personally identifiable household records. Raw sources remain immutable; transformations are script-driven; exact file provenance is documented in the source manifest; and credentials are excluded from version control. Model explanations are labelled as predictive sensitivities, and scenario outputs are labelled as controlled perturbations rather than causal estimates or forecasts. Human review is required before any household, procurement, or policy action.")
    B.add_heading(doc, "Next Steps for the Final Report", 2)
    B.add_bullet(doc, "Acquire release-date-safe arrivals and broaden state-month rural-wage coverage.")
    B.add_bullet(doc, "Expand rolling origins, perform geographic/commodity robustness tests, and calibrate probabilities.")
    B.add_bullet(doc, "Tune stakeholder-specific asymmetric cost rules, especially procurement and policy thresholds.")
    B.add_bullet(doc, "Add area-weighted climate summaries and independent affordability criterion measures.")
    B.add_body(doc, "The final-stage sequence is deliberately ordered. Data-completeness work comes first, followed by longer chronological validation and probability calibration. Only after the predictive layer is stable will stakeholder-specific thresholds be optimized. This avoids tuning an operational rule to an unstable model and preserves the distinction between analytical performance and decision value.")
    B.add_heading(doc, "Interim Conclusion", 2)
    B.add_body(doc, "The aligned evidence chain preserves the synopsis and provides a defensible interim answer to every RQ. Available drivers jointly matter, but the full RQ1 hypothesis remains open until genuine arrivals are integrated. HistGradientBoosting is selected at all horizons, but ML has not yet beaten conventional baselines significantly. HFASI is provisionally construct-valid and perfectly rank-stable across forecast months. Warning value is demonstrably stakeholder-specific: household budgeting and retail inventory improve, whereas current procurement and policy rules do not. The public repository, appendices, and notebook evidence make these conclusions auditable while keeping uncertainty and limitations visible.")
    B.add_page_break(doc)

    # Main report page 16: bibliography.
    references = [
        "Akter, S., & Basher, S. A. (2014). The impacts of food price and income shocks on household food security and economic well-being: Evidence from rural Bangladesh. Global Environmental Change, 25, 150-162. https://doi.org/10.1016/j.gloenvcha.2014.02.003",
        "Bhardwaj, M. R., Pawar, J., Bhat, A., et al. (2023). An innovative deep learning based approach for accurate agricultural crop price prediction. 2023 IEEE International Conference on Automation Science and Engineering. https://doi.org/10.1109/CASE56687.2023.10260494",
        "Cattaneo, A., Sadiddin, A., Vaz, S., et al. (2023). Ensuring affordability of diets in the face of shocks. Food Policy, 117, 102470. https://doi.org/10.1016/j.foodpol.2023.102470",
        "Hyndman, R. J., & Koehler, A. B. (2006). Another look at measures of forecast accuracy. International Journal of Forecasting, 22(4), 679-688. https://doi.org/10.1016/j.ijforecast.2006.03.001",
        "Jain, A., Marvaniya, S., Godbole, S., & Munigala, V. (2020). A framework for crop price forecasting in emerging economies by analyzing the quality of time-series data. arXiv. https://doi.org/10.48550/arXiv.2009.04171",
        "Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. Advances in Neural Information Processing Systems, 30, 4765-4774.",
        "Ma, W., Nowocin, K., Marathe, N., & Chen, G. H. (2019). An interpretable produce price forecasting system for small and marginal farmers in India. Proceedings of ICTD, Article 6, 1-11. https://doi.org/10.1145/3287098.3287100",
        "Madaan, L., Sharma, A., Khandelwal, P., Goel, S., Singla, P., & Seth, A. (2019). Price forecasting and anomaly detection for agricultural commodities in India. COMPASS. https://doi.org/10.1145/3314344.3332488",
        "Makridakis, S., Spiliotis, E., & Assimakopoulos, V. (2018). Statistical and machine learning forecasting methods: Concerns and ways forward. PLOS ONE, 13(3), e0194889. https://doi.org/10.1371/journal.pone.0194889",
        "Malesios, C., Jones, N., & Jones, A. (2020). A change-point analysis of food price shocks. Climate Risk Management, 27, 100208. https://doi.org/10.1016/j.crm.2019.100208",
        "Ministry of Statistics and Programme Implementation. (2025). Household Consumption Expenditure Survey: 2023-24. Government of India.",
        "Open Government Data Platform India. (2026). Current daily price of various commodities from various markets (Mandi). https://www.data.gov.in/catalog/current-daily-price-various-commodities-various-markets-mandi",
        "Theofilou, A., Arvanitakis, G., & Tzouramani, I. (2025). Predicting prices of staple crops using machine learning: A systematic review. Sustainability, 17(12), 5456. https://doi.org/10.3390/su17125456",
        "National Aeronautics and Space Administration. (2026). POWER Data Access Viewer. https://power.larc.nasa.gov/",
        "World Bank. (2026). Commodity markets (Pink Sheet) data. https://www.worldbank.org/en/research/commodity-markets",
    ]
    B.add_heading(doc, "Bibliography", 1)
    for ref in references:
        compact_reference(doc, ref)
    B.add_note(doc, "All official web sources were used through the repository's source manifest and retrieval notebooks. Access and coverage limitations are recorded with the corresponding files.")
    B.add_page_break(doc)

    # Appendix A: repository and reproducibility.
    B.add_heading(doc, "Appendix A", 1)
    B.add_heading(doc, "GitHub Repository Structure and Reproducibility Map", 2)
    B.add_body(doc, "The repository tree below is referenced in the main report and maps evaluator actions to stable public paths.")
    tree = [
        "qm640-food-affordability-ai/",
        "|-- README.md",
        "|-- data/",
        "|   |-- curated/                 # compressed analytical source panels",
        "|   |-- external_required/       # APY, wages, HCES segment aggregates",
        "|   |-- data_dictionary.csv",
        "|   `-- source_manifest.csv",
        "|-- notebooks/",
        "|   |-- 01_data_acquisition_synopsis_aligned.ipynb",
        "|   |-- 02_data_quality_and_cleaning_synopsis_aligned.ipynb",
        "|   |-- 03_exploratory_analysis_synopsis_aligned.ipynb",
        "|   |-- 04_statistical_analysis_synopsis_aligned.ipynb",
        "|   |-- 05_forecasting_models_synopsis_aligned.ipynb",
        "|   |-- 06_shock_classification_synopsis_aligned.ipynb",
        "|   |-- 07_explainability_synopsis_aligned.ipynb",
        "|   |-- 08_affordability_index_synopsis_aligned.ipynb",
        "|   |-- 09_decision_scenario_analysis_synopsis_aligned.ipynb",
        "|   `-- 10_final_results_synopsis_aligned.ipynb",
        "|-- reports/notebook_outputs/    # figures, metrics, summaries, checks",
        "|-- scripts/                     # reusable collection/report utilities",
        "|-- src/                         # reusable Python package",
        "`-- tests/                       # pipeline and quality checks",
    ]
    p = doc.add_paragraph()
    for line in tree:
        r = p.add_run(line + "\n")
        r.font.name = "Courier New"
        r._element.get_or_add_rPr().get_or_add_rFonts().set(qn("w:ascii"), "Courier New")
        r._element.get_or_add_rPr().get_or_add_rFonts().set(qn("w:hAnsi"), "Courier New")
        r.font.size = Pt(8.2)
    B.format_paragraph(p, first_indent=False, align=WD_ALIGN_PARAGRAPH.LEFT, double=False)
    B.add_page_break(doc)
    B.add_heading(doc, "Appendix A (Continued)", 1)
    B.add_table_caption(doc, 8, "Notebook-to-evidence map")
    B.add_table(
        doc,
        ["NB", "Primary purpose", "Verified output"],
        [
            ["01", "Acquisition and source audit", "1,494,745 curated rows; remaining gaps identified"],
            ["02", "Cleaning and integration", "41,792 rows; 1-3 month exact targets"],
            ["03", "EDA", "Volatility, 2/5/10-year pressure, coverage"],
            ["04", "RQ1 inference", "Three fixed-effects models; BH-controlled terms"],
            ["05", "RQ2 forecasting", "12,540 forecasts; H0-2 decisions"],
            ["06", "Shock classification", "Three horizon-specific HGB classifiers"],
            ["07", "Explainability", "Permutation and local sensitivity evidence"],
            ["08", "RQ3 HFASI", "719 HCES aggregates; validity and stability"],
            ["09", "RQ4 decision value", "Four stakeholders; four stress scenarios"],
            ["10", "Consolidation", "155 artifacts; RQ/hypothesis traceability"],
        ],
        [0.8, 2.5, 3.2],
        font_size=7.8,
    )
    B.add_page_break(doc)

    # Appendix B: expanded dictionary.
    B.add_heading(doc, "Appendix B", 1)
    B.add_heading(doc, "Expanded Data Dictionary", 2)
    B.add_table_caption(doc, 9, "Core variables, types, and analytical roles")
    B.add_table(
        doc,
        ["Variable / group", "Type / unit", "Role", "Definition and treatment"],
        [
            ["date, Region, Commodity", "Temporal / categorical", "Panel keys", "Month, state/UT, and essential-food commodity; standardized aliases."],
            ["modal_price, price_mom_pct, price_yoy_pct", "Numeric; INR/unit and %", "Target/predictors", "Observed market price and backward-looking changes; no price interpolation."],
            ["future_price_change_1m/2m/3m", "Numeric; %", "Dependent variables", "Exact calendar forward change at 1, 2, or 3 months."],
            ["shock_1m/2m/3m", "Binary", "Classification targets", "Future change >= training-only commodity-specific 90th percentile."],
            ["rainfall_anomaly_pct", "Numeric; %", "Climate predictor", "Departure from seasonal climatology at representative state point."],
            ["temperature_anomaly_c", "Numeric; C", "Climate predictor", "Temperature departure from climatology."],
            ["relative_humidity_pct", "Numeric; %", "Climate predictor", "Monthly mean relative humidity."],
            ["production, yield", "Numeric; tonnes / yield unit", "Supply predictors", "DES state-crop annual values aligned to commodity and marketing period."],
            ["source_rows", "Count/log", "Quality control", "Contributing official reporting rows; never treated as market arrivals."],
            ["rural_wage", "Numeric", "Macro/HFASI input", "Labour Bureau wage measure; partial match coverage retained."],
            ["food_share, MPCE, decile, sector", "0-1 / INR / category", "HFASI inputs/groups", "HCES 2023-24 disclosure-safe segment aggregates."],
            ["HFASI", "Index centered at 100", "Affordability output", "Food share weighted gap between forecast food-cost and purchasing-power growth."],
        ],
        [1.9, 1.35, 1.35, 2.4],
        font_size=7.2,
    )
    B.add_page_break(doc)

    # Appendix C: formulae and traceability.
    B.add_heading(doc, "Appendix C", 1)
    B.add_heading(doc, "Formulae, Hypothesis Traceability, and Decision Rules", 2)
    B.add_table_caption(doc, 10, "Evaluation formulae and aligned use")
    B.add_table(
        doc,
        ["Measure", "Formula", "Use"],
        [
            ["RMSE", "sqrt[(1/n) SUM(y-yhat)^2]", "Panel forecast loss and model selection"],
            ["Balanced accuracy", "(TPR+TNR)/2", "Rare-shock classification"],
            ["Recall", "TP/(TP+FN)", "Missed-shock sensitivity"],
            ["Average precision", "Area under precision-recall curve", "Rare-event ranking quality"],
            ["Brier score", "(1/n) SUM(p-o)^2", "Probability accuracy"],
            ["Spearman rho", "Correlation of ranked MPCE and food share", "HFASI construct validity"],
            ["Loss reduction", "100 x (baseline loss-model loss)/baseline loss", "Stakeholder decision value"],
            ["HFASI", "100+w_food(g_food-g_power)", "Segment affordability stress"],
        ],
        [1.4, 2.65, 2.45],
        font_size=7.8,
    )
    B.add_table_caption(doc, 11, "Research-question evidence chain")
    B.add_table(
        doc,
        ["RQ", "Primary test/output", "Decision", "Qualification"],
        [
            ["RQ1", "Cluster-robust FE; BH FDR", "Available H0 rejected at 1-3m", "Full H0 awaits arrivals"],
            ["RQ2", "Chronological paired forecast loss", "H0 not rejected at 1-3m", "Only eight aligned origins"],
            ["RQ3", "Spearman + bootstrap + rank stability", "Provisional H0 rejection", "Aggregate segment construct"],
            ["RQ4", "Paired retrospective stakeholder loss", "Partial support", "Cost rules are stakeholder-specific"],
        ],
        [0.6, 2.15, 1.75, 2.0],
        font_size=7.8,
    )
    B.add_page_break(doc)

    # Appendix D: limitations, next-stage plan, and checklist.
    B.add_heading(doc, "Appendix D", 1)
    B.add_heading(doc, "Limitations Register, Next-Stage Work, and Report Checklist", 2)
    B.add_table_caption(doc, 12, "Material limitations and mitigation")
    B.add_table(
        doc,
        ["Limitation", "Impact", "Mitigation / next stage"],
        [
            ["Market arrivals missing", "Full RQ1 hypothesis unresolved", "Acquire official quantity with unit/date lineage."],
            ["Partial rural wages", "Weak purchasing-power panel coverage", "Expand release-safe Labour Bureau harmonization."],
            ["Unbalanced reporting", "Coverage can mimic price change", "Retain source_rows controls and balanced sensitivity samples."],
            ["Point climate locations", "State climate exposure measured imperfectly", "Build area-weighted gridded state summaries."],
            ["Eight forecast origins", "Forecast-loss inference has low power", "Extend chronological backtest as releases accrue."],
            ["Aggregate HCES", "No household-level inference", "Retain disclosure-safe segment language; use authorized microdata if approved."],
            ["Nonlinear scenario extremes", "Probability clips at 1.0", "Treat as stress test; calibrate/extrapolation-check before use."],
            ["Stakeholder loss asymmetry", "Warnings harm procurement/policy under current rules", "Tune threshold and costs separately for each stakeholder."],
            ["Associational evidence", "No causal intervention claim", "Use causal methods only in a separately identified design."],
        ],
        [1.55, 2.15, 2.8],
        font_size=7.5,
    )
    B.add_heading(doc, "Final-Report Checklist", 2)
    B.add_bullet(doc, "Original title, four RQs, and four hypothesis pairs preserved.")
    B.add_bullet(doc, "All mandatory template sections, GitHub links, data dictionary, cleaning log, metrics, preliminary results, limitations, next steps, bibliography, and appendices included.")
    B.add_bullet(doc, "All key numbers transcribed from the aligned execution summaries; no causal overstatement.")
    B.add_bullet(doc, "Main narrative is deliberately page-controlled; appendices hold extended evidence.")
    B.add_page_break(doc)

    # Appendix E: detailed literature relevance matrix.
    B.add_heading(doc, "Appendix E", 1)
    B.add_heading(doc, "Literature Relevance Matrix", 2)
    B.add_body(doc, "This matrix documents how the minimum ten scholarly sources informed the design while keeping the main literature survey interpretive rather than tabular.")
    B.add_table_caption(doc, 13, "Literature relevance matrix")
    B.add_table(
        doc,
        ["Source", "Method / context", "Principal contribution to this study"],
        [
            ["Akter & Basher (2014)", "Food-price shocks and welfare", "Supports unequal household exposure and RQ3 validation."],
            ["Cattaneo et al. (2023)", "Diet-affordability synthesis", "Connects shocks, affordability, and mitigation decisions."],
            ["Hyndman & Koehler (2006)", "Forecast-error measures", "Supports RMSE and scale-aware forecast evaluation."],
            ["Makridakis et al. (2018)", "Statistical versus ML evidence", "Requires strong baselines; complexity is not presumed superior."],
            ["Jain et al. (2020)", "Quality-aware crop forecasting", "Supports reporting-coverage and lineage features."],
            ["Ma et al. (2019)", "Interpretable India price forecasts", "Supports localized forecasts and user-facing explanations."],
            ["Madaan et al. (2019)", "India forecasts and anomalies", "Supports joint regression and shock-classification design."],
            ["Lundberg & Lee (2017)", "Model explanation", "Motivates local/global sensitivity with non-causal caveats."],
            ["Malesios et al. (2020)", "Food-price change points", "Supports explicit shock definitions and structural awareness."],
            ["Theofilou et al. (2025)", "Systematic review", "Identifies generalizability, validation, and exogenous-data gaps."],
        ],
        [1.55, 1.75, 3.2],
        font_size=7.4,
    )
    return doc


def finalize(doc):
    # The retained template's Normal and APA heading styles use 1.15 spacing.
    # Apply that page-efficient rhythm to the report body so the main report
    # remains within the required 15-16 pages before appendices.
    started = False
    in_bibliography = False
    for paragraph in doc.paragraphs:
        text = paragraph.text.strip()
        if text == "GitHub Repository and Project Status":
            started = True
        if not started:
            continue
        if text.startswith("Bibliography"):
            in_bibliography = True
        if text == "Appendix A":
            in_bibliography = False
        style_name = paragraph.style.name if paragraph.style else ""
        if style_name.startswith("Heading"):
            paragraph.paragraph_format.line_spacing = 1.15
            paragraph.paragraph_format.space_after = Pt(5)
            heading_size = 13 if style_name == "Heading 1" else 11.5
            for run in paragraph.runs:
                B.set_run_font(run, size=heading_size, bold=True)
        elif text.startswith("Table ") or text.startswith("Figure "):
            paragraph.paragraph_format.line_spacing = 1.0
            for run in paragraph.runs:
                if run.font.size is None or run.font.size.pt > 9.5:
                    run.font.size = Pt(9.5)
        elif text.startswith("Note."):
            paragraph.paragraph_format.line_spacing = 1.0
            for run in paragraph.runs:
                run.font.size = Pt(8.5)
        elif in_bibliography:
            paragraph.paragraph_format.line_spacing = 1.0
            for run in paragraph.runs:
                run.font.size = Pt(8.2)
        else:
            paragraph.paragraph_format.line_spacing = 1.15
            for run in paragraph.runs:
                if run.font.size is None or run.font.size.pt >= 11:
                    run.font.size = Pt(10.75)

    settings = doc.settings._element
    update_fields = settings.find(qn("w:updateFields"))
    if update_fields is None:
        update_fields = OxmlElement("w:updateFields")
        settings.append(update_fields)
    update_fields.set(qn("w:val"), "true")
    doc.core_properties.title = "QM640 Interim Report - Food Price Affordability AI"
    doc.core_properties.subject = "Synopsis-Aligned Data Analytics Capstone Interim Report"
    doc.core_properties.author = "Piyush Soni"
    doc.core_properties.keywords = (
        "food price, India, forecasting, shock classification, HFASI, explainable AI"
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    doc.save(OUTPUT)


def main():
    if not TEMPLATE.exists():
        raise FileNotFoundError(TEMPLATE)
    doc = build()
    finalize(doc)
    print(f"Created: {OUTPUT}")


if __name__ == "__main__":
    main()
