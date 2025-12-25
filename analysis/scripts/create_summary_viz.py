#!/usr/bin/env python
"""
Create professional summary visualization showing the analysis pathway.
Improved visual design with better spacing, colors, and layout.
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
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output"


def create_summary_visualization() -> Path:
    """
    Create a professional summary visualization with improved visual design.
    
    This visualization shows:
    - The 4-step analysis journey
    - Key visualizations that guided the investigation
    - The final conclusion
    """
    animator.show_info("Creating improved professional analysis pathway summary visualization...")
    
    # Create figure with better proportions
    fig = plt.figure(figsize=(20, 12))
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    # Set background color - soft white
    fig.patch.set_facecolor('#FFFFFF')
    ax.set_facecolor('#FFFFFF')
    
    # Title section - improved typography
    title_y = 7.3
    ax.text(5, 7.6, 'RNA-seq Analysis: From Exploration to Biological Insight', 
            ha='center', va='top', fontsize=22, fontweight='bold', 
            color='#1A1A1A', family='sans-serif')
    ax.text(5, 7.1, 'Evidence-Based Analysis Pathway and Key Findings', 
            ha='center', va='top', fontsize=15, style='italic', 
            color='#5A5A5A', family='sans-serif')
    
    # Step boxes - improved design with better spacing
    step_y = 5.4
    step_height = 1.5
    step_width = 2.1
    step_spacing = 0.25
    
    # Refined color scheme - more professional
    step_colors = {
        1: {'face': '#E3F2FD', 'edge': '#2196F3', 'text': '#1565C0', 'badge': '#1976D2', 'accent': '#BBDEFB'},  # Blue
        2: {'face': '#E8F5E9', 'edge': '#4CAF50', 'text': '#2E7D32', 'badge': '#388E3C', 'accent': '#C8E6C9'},  # Green
        3: {'face': '#FFF9E6', 'edge': '#FFC107', 'text': '#F57C00', 'badge': '#FFA000', 'accent': '#FFECB3'},  # Amber
        4: {'face': '#FCE4EC', 'edge': '#E91E63', 'text': '#C2185B', 'badge': '#C2185B', 'accent': '#F8BBD0'}  # Pink
    }
    
    steps = [
        (1, 'Exploration', 'Both Datasets', 'RNA-seq: clear lead', 0.4),
        (2, 'Hypothesis', 'Population', 'Separation?', 2.75),
        (3, 'Validation', 'Stats + Pathways', '970 genes sig', 5.1),
        (4, 'Conclusion', 'Epithelial', 'Stress Response', 7.45)
    ]
    
    step_boxes = []
    for step_num, title, line1, line2, x_pos in steps:
        colors = step_colors[step_num]
        
        # Main box with subtle shadow effect
        box = FancyBboxPatch(
            (x_pos, step_y), step_width, step_height,
            boxstyle="round,pad=0.25", 
            facecolor=colors['face'],
            edgecolor=colors['edge'],
            linewidth=2.5,
            alpha=0.95
        )
        ax.add_patch(box)
        step_boxes.append((box, x_pos + step_width/2))
        
        # Accent bar at top
        accent_bar = Rectangle((x_pos, step_y + step_height - 0.2), step_width, 0.2,
                              facecolor=colors['accent'], edgecolor='none', zorder=1, alpha=0.8)
        ax.add_patch(accent_bar)
        
        # Step number badge - improved design
        badge = Circle((x_pos + 0.3, step_y + step_height - 0.3), 0.25, 
                      facecolor=colors['badge'], edgecolor='white', linewidth=2.5, zorder=10)
        ax.add_patch(badge)
        ax.text(x_pos + 0.3, step_y + step_height - 0.3, str(step_num), 
               ha='center', va='center', fontsize=14, fontweight='bold', 
               color='white', zorder=11)
        
        # Step title
        ax.text(x_pos + step_width/2, step_y + step_height - 0.55, title, 
               ha='center', va='center', fontsize=14, fontweight='bold',
               color=colors['text'], family='sans-serif')
        
        # Step content - improved spacing
        ax.text(x_pos + step_width/2, step_y + step_height - 0.9, line1, 
               ha='center', va='center', fontsize=11,
               color='#2C3E50', family='sans-serif', fontweight='medium')
        ax.text(x_pos + step_width/2, step_y + step_height - 1.2, line2, 
               ha='center', va='center', fontsize=10, style='italic',
               color='#5A5A5A', family='sans-serif')
    
    # Arrows between steps - improved design
    arrow_y = step_y + step_height/2
    for i in range(len(steps) - 1):
        x_start = steps[i][4] + step_width
        x_end = steps[i+1][4]
        arrow = FancyArrowPatch(
            (x_start + 0.05, arrow_y), (x_end - 0.05, arrow_y),
            arrowstyle='->', mutation_scale=25, 
            linewidth=3, color='#90A4AE',
            zorder=5, alpha=0.8,
            connectionstyle="arc3,rad=0"
        )
        ax.add_patch(arrow)
    
    # Key Visualizations section - improved design
    viz_section_y = 3.4
    ax.text(5, 3.95, 'Key Visualizations That Guided the Investigation', 
            ha='center', va='center', fontsize=14, fontweight='bold',
            color='#1A1A1A', family='sans-serif')
    
    viz_y = 2.4
    viz_height = 1.0
    viz_width = 1.75
    viz_spacing = 0.15
    
    # Refined visualization colors
    visualizations = [
        ('RNA-seq PCA', 'Clear structure', '2 populations', '#2196F3', '#E3F2FD', 0.25),  # Blue
        ('Diabetes PCA', 'No clear lead', 'Diffuse pattern', '#E91E63', '#FCE4EC', 2.15),  # Pink
        ('Volcano Plot', 'Statistical', 'validation', '#FF5722', '#FFEBEE', 4.05),  # Deep orange
        ('Pathway', 'Biological', 'validation', '#4CAF50', '#E8F5E9', 5.95),  # Green
        ('Top Genes', 'Keratin genes', 'identified', '#FF9800', '#FFF3E0', 7.85)  # Orange
    ]
    
    for viz_title, line1, line2, accent_color, pastel_color, x_pos in visualizations:
        # Main box with improved design
        viz_box = FancyBboxPatch(
            (x_pos, viz_y), viz_width, viz_height,
            boxstyle="round,pad=0.18",
            facecolor='white',
            edgecolor=accent_color,
            linewidth=2.5,
            alpha=0.98
        )
        ax.add_patch(viz_box)
        
        # Accent bar at top - more prominent
        accent = Rectangle((x_pos, viz_y + viz_height - 0.18), viz_width, 0.18,
                          facecolor=pastel_color, edgecolor='none', zorder=1, alpha=0.7)
        ax.add_patch(accent)
        
        # Title - improved typography
        ax.text(x_pos + viz_width/2, viz_y + viz_height - 0.4, viz_title,
               ha='center', va='center', fontsize=12, fontweight='bold',
               color=accent_color, family='sans-serif')
        
        # Content lines - better spacing
        ax.text(x_pos + viz_width/2, viz_y + viz_height - 0.65, line1,
               ha='center', va='center', fontsize=10, fontweight='medium',
               color='#2C3E50', family='sans-serif')
        ax.text(x_pos + viz_width/2, viz_y + viz_height - 0.85, line2,
               ha='center', va='center', fontsize=9, style='italic',
               color='#5A5A5A', family='sans-serif')
    
    # Key Finding section - improved design
    finding_y = 0.3
    finding_height = 1.1
    finding_width = 9.2
    finding_x = 0.4
    
    # Main finding box with gradient effect
    finding_box = FancyBboxPatch(
        (finding_x, finding_y), finding_width, finding_height,
        boxstyle="round,pad=0.25",
        facecolor='#FFF8E1',  # Soft amber
        edgecolor='#FFC107',
        linewidth=3,
        alpha=0.98
    )
    ax.add_patch(finding_box)
    
    # Header bar - more prominent
    header_bar = Rectangle((finding_x, finding_y + finding_height - 0.25), 
                          finding_width, 0.25,
                          facecolor='#FFC107', edgecolor='none', zorder=1, alpha=0.85)
    ax.add_patch(header_bar)
    
    # Key Finding title - improved
    ax.text(finding_x + finding_width/2, finding_y + finding_height - 0.12,
           'Key Finding',
           ha='center', va='center', fontsize=16, fontweight='bold',
           color='#E65100', family='sans-serif', zorder=2)
    
    # Main finding text - improved typography
    ax.text(finding_x + finding_width/2, finding_y + finding_height - 0.5,
           'Two populations separated by epithelial activation',
           ha='center', va='center', fontsize=14, fontweight='bold',
           color='#1A1A1A', family='sans-serif')
    
    # Sub-finding text - improved
    ax.text(finding_x + finding_width/2, finding_y + finding_height - 0.8,
           'ADCs/Free drugs induce keratinization stress response',
           ha='center', va='center', fontsize=12, style='italic',
           color='#424242', family='sans-serif')
    
    plt.tight_layout()
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "analysis_pathway_summary.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='#FFFFFF', 
                edgecolor='none', pad_inches=0.25)
    plt.close()
    
    animator.show_file_operation("Saved improved analysis pathway summary", str(output_path))
    return output_path


if __name__ == "__main__":
    create_summary_visualization()
