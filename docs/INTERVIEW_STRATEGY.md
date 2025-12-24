# Interview Exercise Strategy Guide

Based on research and best practices for Hamilton DAG interviews.

## 🎯 What They're Evaluating

From the README, they're looking for:

1. **AI Collaboration** - Your prompting strategy, iteration process, knowing when to trust/question AI
2. **Analytical Thinking** - How you explore data, form hypotheses, validate findings
3. **Communication** - Talking through your reasoning as you work

**NOT testing**: Memorized knowledge, perfect answers, trick questions

## 📋 Best Practices from Research

### 1. **Start with Clear Objectives**
- Define specific questions you want to answer
- Focus on meaningful insights, not just data processing
- Example questions:
  - "What factors predict readmission <30 days?"
  - "How do ADCs compare to free drugs in gene expression?"
  - "What's the relationship between medications and readmission?"

### 2. **Prioritize Data Quality**
- Handle missing values (encoded as `?`)
- Validate data before analysis
- Document data quality issues you find
- Show you understand the messiness

### 3. **Design Modular, Scalable Pipeline**
- Create focused analysis nodes (one question per node)
- Build incrementally (simple → complex)
- Use Hamilton's DAG structure effectively
- Show clear data lineage

### 4. **Demonstrate Analytical Thinking**
- Explore data first (you already did this!)
- Form hypotheses before coding
- Validate findings with multiple approaches
- Show your reasoning process

### 5. **Leverage Hamilton's Strengths**
- Use function parameters to create clear DAG edges
- Build reusable transformation functions
- Use `@_cached` decorator for expensive operations
- Show understanding of DAG concepts

## 🚀 Recommended Next Steps

### Phase 1: Formulate Questions (5-10 min)

**Diabetes Dataset Questions:**
1. What demographic factors predict readmission?
2. How does time in hospital relate to readmission?
3. Which medications are associated with readmission?
4. What's the relationship between lab procedures and readmission?

**RNA-seq Dataset Questions:**
1. How do ADCs compare to free drugs in gene expression?
2. What genes are differentially expressed by concentration?
3. Are there compound-specific gene signatures?
4. How do controls compare to treatments?

**Pick 2-3 questions** that interest you and are answerable in ~1 hour.

### Phase 2: Build Analysis Nodes (30-40 min)

**Start Simple:**
1. Create one analysis node for your first question
2. Run it and validate the results
3. Build on it (add more nodes)
4. Connect nodes in the DAG

**Example Pattern:**
```python
# Simple analysis
@_cached
def readmission_by_gender(raw_diabetic_data: pd.DataFrame) -> pd.DataFrame:
    """Readmission rates by gender."""
    return raw_diabetic_data.groupby('gender')['readmitted'].value_counts(normalize=True)

# More complex analysis (builds on simple)
@_cached
def readmission_by_gender_and_age(raw_diabetic_data: pd.DataFrame) -> pd.DataFrame:
    """Readmission rates by gender and age."""
    return raw_diabetic_data.groupby(['gender', 'age'])['readmitted'].value_counts(normalize=True)
```

### Phase 3: Validate and Document (10-15 min)

- Check your results make sense
- Document any interesting findings
- Note data quality issues you handled
- Explain your reasoning

## 💡 Key Tips

### For AI Collaboration
- **Show your prompts** - Explain what you asked AI and why
- **Iterate thoughtfully** - Show how you refined your approach
- **Question AI output** - Validate code before using it
- **Explain decisions** - Why did you choose this approach?

### For Analytical Thinking
- **Explore before coding** - Understand data structure first
- **Form hypotheses** - "I think X because Y"
- **Validate findings** - Check multiple ways
- **Handle edge cases** - Missing data, outliers, etc.

### For Communication
- **Talk through reasoning** - Explain your thought process
- **Document decisions** - Why this approach?
- **Show trade-offs** - What did you consider?
- **Be honest** - "I tried X, it didn't work, so I did Y"

## 🎯 Recommended Starting Point

**Option 1: Diabetes - Readmission Analysis**
- Start with `readmission_by_gender` (simple)
- Build to `readmission_by_gender_and_time_in_hospital` (complex)
- Show progression of thinking

**Option 2: RNA-seq - Compound Comparison**
- Start with `gene_expression_by_compound` (simple)
- Build to `differential_expression_adc_vs_free` (complex)
- Show biological insights

**Option 3: Cross-Dataset Insights**
- If time allows, connect insights between datasets
- Shows advanced thinking

## 📊 Success Criteria

You'll succeed if you:
- ✅ Show clear analytical thinking
- ✅ Build working Hamilton nodes
- ✅ Handle data quality issues
- ✅ Communicate your reasoning
- ✅ Demonstrate AI collaboration skills
- ✅ Create meaningful insights

**You DON'T need:**
- ❌ Perfect answers
- ❌ Complex ML models
- ❌ All questions answered
- ❌ Production-ready code

## 🚦 Time Management

- **0-10 min**: Formulate questions, explore data
- **10-50 min**: Build 2-3 analysis nodes
- **50-60 min**: Validate, document, communicate findings

## 🎓 Remember

This is about **how you think and work**, not what you know. Show your process, explain your reasoning, and demonstrate that you can:
- Learn new frameworks (Hamilton)
- Work with messy data
- Collaborate with AI effectively
- Think analytically
- Communicate clearly

Good luck! 🚀

