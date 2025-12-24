#!/usr/bin/env python
"""
Generate PDF report with embedded figures.

Creates a professional PDF document from the analysis report with all figures embedded.
"""

import sys
from pathlib import Path
from datetime import datetime

# Workspace import setup
workspace_root = Path(__file__).parent.parent.parent.parent.parent
if str(workspace_root) not in sys.path:
    sys.path.insert(0, str(workspace_root))
from workspace.imports import setup_workspace_imports
setup_workspace_imports(__file__)

from utils.animation_utils import animator
from utils.debug_utils import quick_debug_setup

logger, config = quick_debug_setup(__name__)

try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak, Table, TableStyle
    from reportlab.lib import colors
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from PIL import Image as PILImage
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False
    animator.show_warning("reportlab not installed. Install with: pip install reportlab")

PROJECT_ROOT = Path(__file__).parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output"


def get_image_dimensions(img_path: Path, max_width: float = 7.0) -> tuple[float, float]:
    """
    Get image dimensions preserving aspect ratio.
    
    Args:
        img_path: Path to image file
        max_width: Maximum width in inches
        
    Returns:
        Tuple of (width, height) in inches
    """
    try:
        with PILImage.open(img_path) as img:
            # Get actual dimensions
            img_width_px, img_height_px = img.size
            aspect_ratio = img_width_px / img_height_px
            
            # Calculate dimensions preserving aspect ratio
            width_inches = min(max_width, 7.0)
            height_inches = width_inches / aspect_ratio
            
            # Ensure it fits on page (max height ~9 inches for letter size)
            if height_inches > 9.0:
                height_inches = 9.0
                width_inches = height_inches * aspect_ratio
            
            return (width_inches, height_inches)
    except Exception as e:
        logger.warning(f"Could not get image dimensions for {img_path}: {e}")
        # Default fallback
        return (7.0, 5.0)


def generate_pdf_report():
    """Generate PDF report with embedded figures."""
    if not REPORTLAB_AVAILABLE:
        animator.show_error("reportlab not installed. Install with: pip install reportlab")
        return None
    
    animator.show_operation_start("Generating PDF Report", "Creating professional PDF with embedded figures")
    
    output_path = PROJECT_ROOT / "FINAL_REPORT.pdf"
    
    # Create PDF document
    doc = SimpleDocTemplate(str(output_path), pagesize=letter,
                           rightMargin=50, leftMargin=50,
                           topMargin=50, bottomMargin=50)
    
    # Container for the 'Flowable' objects
    story = []
    
    # Define styles
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#1a1a1a'),
        spaceAfter=30,
        alignment=1  # Center
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        textColor=colors.HexColor('#2c3e50'),
        spaceAfter=12,
        spaceBefore=12
    )
    
    # Title
    story.append(Paragraph("RNA-seq Drug Screening Analysis", title_style))
    story.append(Paragraph("Final Report", styles['Normal']))
    story.append(Spacer(1, 0.2*inch))
    story.append(Paragraph(f"<i>Generated: {datetime.now().strftime('%B %d, %Y')}</i>", styles['Normal']))
    story.append(Spacer(1, 0.3*inch))
    
    # Executive Summary
    story.append(Paragraph("Executive Summary", heading_style))
    story.append(Paragraph(
        "Analysis of 52 RNA-seq samples revealed two distinct populations based on gene expression patterns. "
        "The separation is driven by epithelial activation and keratinization pathways, indicating that ADCs "
        "and free cytotoxic drugs induce a stress response in skin organoids.",
        styles['Normal']
    ))
    story.append(Spacer(1, 0.1*inch))
    story.append(Paragraph(
        "<b>Validation:</b> Statistical (970 genes significantly different, p < 0.05), Pathway Enrichment "
        "(20 significant pathways, FDR < 0.05), and Literature (keratin upregulation is a known marker of "
        "epithelial stress/repair).",
        styles['Normal']
    ))
    story.append(Spacer(1, 0.3*inch))
    
    # Add summary visualization
    summary_viz_path = OUTPUT_DIR / "analysis_pathway_summary.png"
    if not summary_viz_path.exists():
        from analysis.scripts.create_summary_viz import create_summary_visualization
        summary_viz_path = create_summary_visualization()
    
    if summary_viz_path and summary_viz_path.exists():
        story.append(Paragraph("Analysis Pathway Summary", heading_style))
        story.append(Paragraph(
            "The analysis journey progressed from initial exploration through hypothesis formation, statistical validation, "
            "and biological interpretation. The key visualizations that guided the investigation are highlighted below.",
            styles['Normal']
        ))
        story.append(Spacer(1, 0.1*inch))
        # Get actual dimensions preserving aspect ratio
        img_width, img_height = get_image_dimensions(summary_viz_path, max_width=7.5)
        img = Image(str(summary_viz_path), width=img_width*inch, height=img_height*inch)
        story.append(img)
        story.append(Spacer(1, 0.3*inch))
        story.append(PageBreak())
    
    # Add ALL figures with article-style explanations
    figure_data = [
        ("Principal Component Analysis", 
         OUTPUT_DIR / "exploratory" / "pca_plot.png",
         "Principal Component Analysis revealed clear separation into two clusters along PC1 (25.0% variance explained). "
         "This was the first indication that the samples formed distinct populations. The separation did not immediately "
         "align with compound categories, raising questions about the underlying biological drivers of this pattern. "
         "PC2 explains an additional 10.4% of variance, further supporting the presence of two distinct groups."),
        
        ("Sample Correlation Matrix", 
         OUTPUT_DIR / "exploratory" / "correlation_heatmap.png",
         "The sample correlation matrix shows high correlation within each population, confirming the clustering observed "
         "in PCA. Samples within Population 1 (controls/Dexamethasone) show strong correlation with each other, as do "
         "samples within Population 2 (ADCs/free drugs). The clear block structure in the heatmap validates the "
         "population separation identified through PCA."),
        
        ("Compound Expression Comparison", 
         OUTPUT_DIR / "exploratory" / "compound_comparison_heatmap.png",
         "Expression patterns across compounds reveal distinct clustering. The top 50 most variable genes show clear "
         "separation between control compounds and cytotoxic drugs. This heatmap visualizes the expression differences "
         "that drive the population separation, with keratin genes and epithelial markers showing the strongest "
         "differential expression."),
        
        ("Volcano Plot - Statistical Validation", 
         OUTPUT_DIR / "population_analysis" / "statistical_tests" / "volcano_plot.png",
         "The volcano plot displays statistical significance (-log10 p-value) versus fold change. Analysis identified "
         "970 genes significantly different between populations (p < 0.05). Most significant genes show positive fold "
         "change, indicating upregulation in Population 2. This confirmed the separation was statistically robust and "
         "not due to noise. Both parametric (t-test) and non-parametric (Mann-Whitney U) tests agreed, with strong "
         "effect sizes (Cohen's d > 1.0) for top genes."),
        
        ("Top Separating Genes - Expression Patterns", 
         OUTPUT_DIR / "population_analysis" / "differential_expression" / "population_separation_top_genes.png",
         "Expression of the top 20 genes separating the populations shows clear differences between Population 1 and "
         "Population 2. All top genes are epithelial structural proteins involved in keratin filament formation, "
         "desmosome assembly, and cornified envelope formation. The box plots demonstrate consistent upregulation in "
         "Population 2 across all top genes, confirming the epithelial activation pattern."),
        
        ("Feature Importance Ranking", 
         OUTPUT_DIR / "population_analysis" / "feature_importance" / "feature_importance_top_genes.png",
         "The top 30 genes ranked by absolute expression difference between populations. All top genes (KRT1, KRT5, "
         "KRT6A, DSG1, PKP1, etc.) show higher expression in Population 2, confirming the epithelial activation pattern. "
         "Genes are colored by direction of change: red indicates higher expression in Population 2, blue indicates higher "
         "expression in Population 1. The consistent pattern of keratin and desmosome genes at the top validates the "
         "biological interpretation."),
        
        ("Pathway Enrichment Analysis", 
         OUTPUT_DIR / "pathway_enrichment" / "pathway_enrichment_top_pathways.png",
         "Pathway enrichment analysis identified 20 significant pathways (FDR < 0.05) related to keratinization and "
         "epithelial barrier function. The most significant pathway was 'Keratinization' (FDR = 6.48e-26). This validated "
         "the hypothesis that Population 2 represents an activated stress response state. Literature search confirmed "
         "that keratin upregulation is a known marker of epithelial stress response, skin barrier repair, and cytotoxic "
         "drug response, providing confidence in the biological interpretation."),
        
        ("Population Summary", 
         OUTPUT_DIR / "visualizations" / "population_summary.png",
         "Summary visualization showing population separation in PCA space (left panel) and sample distribution by compound "
         "(right panel). The PCA plot clearly shows the two populations with minimal overlap. The compound distribution "
         "reveals that controls and Dexamethasone treatments cluster in Population 1, while ADCs and free drugs cluster "
         "in Population 2, supporting the biological interpretation of baseline versus stress response states."),
    ]
    
    for title, fig_path, explanation in figure_data:
        if fig_path.exists():
            story.append(Paragraph(title, heading_style))
            # Clean explanation and make article-style
            explanation_clean = explanation.replace('&lt;', '<').replace('&gt;', '>')
            story.append(Paragraph(explanation_clean, styles['Normal']))
            story.append(Spacer(1, 0.15*inch))
            
            # Get actual dimensions preserving aspect ratio
            img_width, img_height = get_image_dimensions(fig_path, max_width=7.0)
            img = Image(str(fig_path), width=img_width*inch, height=img_height*inch)
            story.append(img)
            story.append(Spacer(1, 0.3*inch))
            story.append(PageBreak())
    
    # Add Conclusions section
    story.append(Paragraph("Conclusions", heading_style))
    story.append(Paragraph(
        "<b>Key Finding:</b> Two distinct populations separated by epithelial activation. ADCs and free drugs "
        "induce keratinization stress response in skin organoids.",
        styles['Normal']
    ))
    story.append(Spacer(1, 0.15*inch))
    story.append(Paragraph(
        "<b>Biological Interpretation:</b> Population 1 (Controls/Dexamethasone) represents baseline epithelial state. "
        "Population 2 (ADCs/Free Drugs) represents activated stress response state with upregulation of keratinization "
        "pathways, epithelial barrier remodeling, and tissue repair mechanisms.",
        styles['Normal']
    ))
    story.append(Spacer(1, 0.15*inch))
    story.append(Paragraph(
        "<b>Implications:</b> Keratin genes could serve as toxicity biomarkers. The epithelial stress response is expected "
        "for cytotoxic compounds, and the tissue-level response indicates the organoid model is working as intended.",
        styles['Normal']
    ))
    
    # Build PDF
    doc.build(story)
    
    animator.show_operation_complete("PDF Report Generation", 0)
    animator.show_file_operation("Generated PDF report", str(output_path))
    
    return output_path


if __name__ == "__main__":
    generate_pdf_report()
