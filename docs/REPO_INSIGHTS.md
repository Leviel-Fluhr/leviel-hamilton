# Key Insights from Repository Documentation

## Direct Quotes from pzl-interview-template/README.md

### What They're Evaluating (Exact Quote)

> "This is **not** a test of what you know. We're interested in:
> 
> 1. **How you collaborate with AI** - Your prompting strategy, iteration process, and knowing when to trust (or question) AI output
> 2. **Your analytical thinking** - How you explore data, form hypotheses, and validate findings
> 3. **Communication** - Talking through your reasoning as you work
> 
> There are no trick questions. The datasets are messy. Perfect answers don't exist."

### Key Points from Repo

1. **Time**: ~1 hour
2. **Tools**: This repo + AI assistant (Claude Code or similar)
3. **Datasets are messy**: Real-world data with real-world problems
4. **No perfect answers**: They want to see your process, not perfection

### Data Quality Notes (From Repo)

**Diabetes:**
- Missing values encoded as `?` (not NaN)
- High missingness in `weight`, `medical_specialty`, `payer_code`
- Class imbalance in readmission target (~54% No, ~35% >30, ~11% <30)
- Diagnosis codes are raw ICD-9

**RNA-seq:**
- 52 samples across 3 plates
- 14 compounds (ADCs, free drugs, controls)
- 78,932 genes (Salmon quantification)

### Hamilton Pattern (From Repo)

The repo shows the pattern clearly:
- Functions become DAG nodes
- Parameter names matching function names create edges
- Use `@_cached` decorator for caching
- Example: `readmission_by_age(raw_diabetic_data: pd.DataFrame)` creates edge from `raw_diabetic_data` → `readmission_by_age`

### What This Means

1. **They expect you to use AI** - It's part of the evaluation
2. **They want to see your process** - Not just results
3. **Messy data is intentional** - Shows how you handle real-world problems
4. **Time is limited** - Focus on 2-3 meaningful analyses, not everything
5. **Communication matters** - Explain your thinking as you go

## Combined Strategy

**From Repo + Web Research:**

1. ✅ **Start with exploration** (you did this!)
2. ✅ **Form hypotheses** (pick 2-3 questions)
3. ✅ **Build incrementally** (simple → complex)
4. ✅ **Handle data quality** (missing values, etc.)
5. ✅ **Document your process** (show AI collaboration)
6. ✅ **Communicate reasoning** (explain decisions)

## What Success Looks Like

Based on repo + research:

- ✅ You explored the data thoughtfully
- ✅ You formed clear hypotheses
- ✅ You built working Hamilton nodes
- ✅ You handled data quality issues
- ✅ You showed your AI collaboration process
- ✅ You communicated your reasoning
- ✅ You created meaningful insights (even if simple)

**You DON'T need:**
- ❌ Complex ML models
- ❌ All questions answered
- ❌ Perfect code
- ❌ Production-ready solutions

## Bottom Line

The repo is very clear: **This is about HOW you work, not WHAT you know.**

Show your process, collaborate with AI effectively, think analytically, and communicate clearly. That's what they're evaluating.

