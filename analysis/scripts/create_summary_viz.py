#!/usr/bin/env python
"""
Create professional summary visualization showing the analysis pathway.
Uses pastel colors and clean design.
"""

import sys
from pathlib import Path

# Workspace import setup
workspace_root = Path(__file__).parent.parent.parent.parent.parent
if str(workspace_root) not in sys.path:
    sys.path.insert(0, str(workspace_root))
from workspace.imports import setup_workspace_imports
setup_workspace_imports(__file__)

from utils.animation_utils import animator
from utils.debug_utils import quick_debug_setup

logger, config = quick_debug_setup(__name__)

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output"


def create_summary_visualization() -> Path:
    """
    Create a professional summary visualization with pastel colors and clean design.
    
    This visualization shows:
    - The 4-step analysis journey
    - Key visualizations that guided the investigation
    - The final conclusion
    """
    animator.show_info("Creating professional analysis pathway summary visualization with pastel colors...")
    
    # Create figure with better proportions
    fig = plt.figure(figsize=(18, 11))
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    # Set background color - soft white/cream
    fig.patch.set_facecolor('#FEFEFE')
    ax.set_facecolor('#FEFEFE')
    
    # Title section - softer colors
    title_y = 7.2
    ax.text(5, 7.5, 'RNA-seq Analysis: From Exploration to Biological Insight', 
            ha='center', va='top', fontsize=20, fontweight='bold', 
            color='#2C3E50', family='sans-serif')
    ax.text(5, 7.0, 'Analysis Pathway and Key Findings', 
            ha='center', va='top', fontsize=14, style='italic', 
            color='#7F8C8D', family='sans-serif')
    
    # Step boxes - pastel color scheme
    step_y = 5.2
    step_height = 1.4
    step_width = 1.9
    step_spacing = 0.3
    
    # Pastel color scheme - soft and clean
    step_colors = {
        1: {'face': '#E8F4F8', 'edge': '#85C1E2', 'text': '#2874A6', 'badge': '#5DADE2'},  # Soft blue
        2: {'face': '#E8F8E8', 'edge': '#82E0AA', 'text': '#27AE60', 'badge': '#52BE80'},  # Soft green
        3: {'face': '#FFF8E7', 'edge': '#F7DC6F', 'text': '#D68910', 'badge': '#F4D03F'},  # Soft yellow
        4: {'face': '#FADBD8', 'edge': '#F1948A', 'text': '#C0392B', 'badge': '#E74C3C'}  # Soft pink/rose
    }
    
    steps = [
        (1, 'Exploration', 'PCA Analysis', 'Found 2 clusters', 0.6),
        (2, 'Hypothesis', 'Population', 'Separation?', 2.8),
        (3, 'Validation', 'Stats + Pathways', '970 genes sig', 5.0),
        (4, 'Conclusion', 'Epithelial', 'Stress Response', 7.2)
    ]
    
    step_boxes = []
    for step_num, title, line1, line2, x_pos in steps:
        colors = step_colors[step_num]
        # Softer rounded box
        box = FancyBboxPatch(
            (x_pos, step_y), step_width, step_height,
            boxstyle="round,pad=0.2", 
            facecolor=colors['face'],
            edgecolor=colors['edge'],
            linewidth=2.0,
            alpha=0.9
        )
        ax.add_patch(box)
        step_boxes.append((box, x_pos + step_width/2))
        
        # Step number badge - softer
        badge = plt.Circle((x_pos + 0.25, step_y + step_height - 0.25), 0.22, 
                          facecolor=colors['badge'], edgecolor='white', linewidth=2, zorder=10, alpha=0.95)
        ax.add_patch(badge)
        ax.text(x_pos + 0.25, step_y + step_height - 0.25, str(step_num), 
               ha='center', va='center', fontsize=13, fontweight='bold', 
               color='white', zorder=11)
        
        # Step title
        ax.text(x_pos + step_width/2, step_y + step_height - 0.5, title, 
               ha='center', va='center', fontsize=13, fontweight='bold',
               color=colors['text'], family='sans-serif')
        
        # Step content
        ax.text(x_pos + step_width/2, step_y + step_height - 0.85, line1, 
               ha='center', va='center', fontsize=10,
               color='#34495E', family='sans-serif')
        ax.text(x_pos + step_width/2, step_y + step_height - 1.15, line2, 
               ha='center', va='center', fontsize=9, style='italic',
               color='#7F8C8D', family='sans-serif')
    
    # Arrows between steps - softer pastel gray
    arrow_y = step_y + step_height/2
    for i in range(len(steps) - 1):
        x_start = steps[i][4] + step_width
        x_end = steps[i+1][4]
        arrow = FancyArrowPatch(
            (x_start, arrow_y), (x_end, arrow_y),
            arrowstyle='->', mutation_scale=22, 
            linewidth=2.5, color='#BDC3C7',
            zorder=5, alpha=0.7
        )
        ax.add_patch(arrow)
    
    # Key Visualizations section - pastel colors
    viz_section_y = 3.2
    ax.text(5, 3.8, 'Key Visualizations That Guided the Investigation', 
            ha='center', va='center', fontsize=13, fontweight='bold',
            color='#34495E', family='sans-serif')
    
    viz_y = 2.3
    viz_height = 0.9
    viz_width = 2.1
    viz_spacing = 0.25
    
    # Pastel visualization colors
    visualizations = [
        ('PCA Plot', 'Showed clear', '2 populations', '#AED6F1', '#3498DB', 0.5),  # Soft blue
        ('Volcano Plot', 'Statistical', 'validation', '#F5B7B1', '#E74C3C', 2.9),  # Soft red
        ('Pathway', 'Biological', 'validation', '#A9DFBF', '#27AE60', 5.3),  # Soft green
        ('Top Genes', 'Keratin genes', 'identified', '#F9E79F', '#F39C12', 7.7)  # Soft orange
    ]
    
    for viz_title, line1, line2, pastel_color, accent_color, x_pos in visualizations:
        # Main box - very soft pastel
        viz_box = FancyBboxPatch(
            (x_pos, viz_y), viz_width, viz_height,
            boxstyle="round,pad=0.15",
            facecolor='white',
            edgecolor=pastel_color,
            linewidth=2.0,
            alpha=0.95
        )
        ax.add_patch(viz_box)
        
        # Pastel accent bar at top
        accent = Rectangle((x_pos, viz_y + viz_height - 0.15), viz_width, 0.15,
                          facecolor=pastel_color, edgecolor='none', zorder=1, alpha=0.6)
        ax.add_patch(accent)
        
        # Title
        ax.text(x_pos + viz_width/2, viz_y + viz_height - 0.35, viz_title,
               ha='center', va='center', fontsize=11, fontweight='bold',
               color=accent_color, family='sans-serif')
        
        # Content lines
        ax.text(x_pos + viz_width/2, viz_y + viz_height - 0.6, line1,
               ha='center', va='center', fontsize=9,
               color='#34495E', family='sans-serif')
        ax.text(x_pos + viz_width/2, viz_y + viz_height - 0.8, line2,
               ha='center', va='center', fontsize=8, style='italic',
               color='#7F8C8D', family='sans-serif')
    
    # Key Finding section - soft pastel design
    finding_y = 0.4
    finding_height = 1.0
    finding_width = 9.0
    finding_x = 0.5
    
    # Soft pastel finding box
    finding_box = FancyBboxPatch(
        (finding_x, finding_y), finding_width, finding_height,
        boxstyle="round,pad=0.2",
        facecolor='#FEF5E7',  # Very soft cream/peach
        edgecolor='#F8C471',
        linewidth=2.5,
        alpha=0.95
    )
    ax.add_patch(finding_box)
    
    # Soft pastel header bar
    header_bar = Rectangle((finding_x, finding_y + finding_height - 0.2), 
                          finding_width, 0.2,
                          facecolor='#F8C471', edgecolor='none', zorder=1, alpha=0.7)
    ax.add_patch(header_bar)
    
    # Key Finding title
    ax.text(finding_x + finding_width/2, finding_y + finding_height - 0.1,
           'Key Finding',
           ha='center', va='center', fontsize=15, fontweight='bold',
           color='#7D6608', family='sans-serif', zorder=2)
    
    # Main finding text
    ax.text(finding_x + finding_width/2, finding_y + finding_height - 0.5,
           'Two populations separated by epithelial activation',
           ha='center', va='center', fontsize=13, fontweight='bold',
           color='#2C3E50', family='sans-serif')
    
    # Sub-finding text
    ax.text(finding_x + finding_width/2, finding_y + finding_height - 0.75,
           'ADCs/Free drugs induce keratinization stress response',
           ha='center', va='center', fontsize=11, style='italic',
           color='#566573', family='sans-serif')
    
    # Remove grid lines for cleaner look
    
    plt.tight_layout()
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "analysis_pathway_summary.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='#FEFEFE', 
                edgecolor='none', pad_inches=0.2)
    plt.close()
    
    animator.show_file_operation("Saved improved analysis pathway summary", str(output_path))
    return output_path


if __name__ == "__main__":
    create_summary_visualization()
