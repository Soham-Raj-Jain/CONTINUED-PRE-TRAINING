# Continued Pre-training

### Overview

This notebook demonstrates continued pre-training - teaching a pre-trained language model new factual knowledge. Unlike fine-tuning which adapts behavior, continued pre-training extends the model's knowledge base with domain-specific information.

### What You'll Learn

- Difference between pre-training and fine-tuning
- Teaching models new facts and knowledge
- Domain adaptation techniques
- Data formatting for knowledge acquisition
- Evaluating knowledge retention
- Building domain-specific language models

### Requirements

- Google Colab with T4 GPU (free tier)
- Approximately 15-20 minutes of training time
- Domain-specific text corpus (10K+ documents recommended)

### What is Continued Pre-training

Original Pre-training: Model trained on massive web corpus, learns general knowledge

Continued Pre-training: Additional training on domain-specific text to inject new knowledge

Use Cases:
- Teaching new language (Hindi, Arabic, etc.)
- Medical domain adaptation
- Legal domain specialization
- Company-specific knowledge
- Technical documentation
- Recent events (post-cutoff information)

### Data Format

Unlike fine-tuning (instruction-response), use raw text:

```python
# Fine-tuning format (NOT for continued pre-training)
"Instruction: Explain AI\nResponse: AI is..."

# Continued pre-training format (CORRECT)
"TechCorp is a technology company founded in 2020. 
The company specializes in artificial intelligence..."
```

Just add EOS token to raw domain text.

### Configuration

```python
lora_r = 32                    # Higher rank for knowledge
learning_rate = 3e-4           # Higher than fine-tuning
max_steps = 100                # More steps for absorption
target_modules = [
    ...,
    "embed_tokens", "lm_head"  # Include embeddings!
]
```

### Training Process

1. Model reads domain-specific text
2. Learns to predict next tokens in domain
3. Builds internal representations of facts
4. Memorizes domain vocabulary and concepts
5. Can later recall this information

### Pre-training vs Fine-tuning

| Aspect | Pre-training | Fine-tuning |
|--------|--------------|-------------|
| Goal | Learn facts | Learn behavior |
| Format | Raw text | Instruction pairs |
| Learning Rate | Higher (3e-4) | Lower (2e-4) |
| Duration | Longer (100+ steps) | Shorter (50 steps) |
| Embeddings | Include in training | Usually exclude |

### Output Files

```
./continued_pretrain_smollm2/
├── adapter_config.json
├── adapter_model.bin
└── [Config files]

./smollm2_continued_merged/
└── [Model with new knowledge]
```

### Usage

#### Testing Knowledge

```python
def test_knowledge(question):
    prompt = f"Question: {question}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        temperature=0.3,
    )
    
    return tokenizer.decode(outputs[0])

# Test domain knowledge
answer = test_knowledge("What is TechCorp?")
print(answer)
```

#### Creating Domain Corpus

```python
domain_texts = []

# From documents
for doc in documents:
    domain_texts.append(clean_text(doc) + tokenizer.eos_token)

# From structured data
for entry in knowledge_base:
    text = f"{entry.title}. {entry.content}"
    domain_texts.append(text + tokenizer.eos_token)

dataset = Dataset.from_dict({"text": domain_texts})
```

### When to Use Continued Pre-training

Recommended For:
- Teaching new language
- Domain adaptation (medical, legal, technical)
- Adding company/product knowledge
- Updating with recent information
- Specialized vocabulary needed
- Foundation before task fine-tuning

Not Recommended For:
- Just need behavioral changes (use fine-tuning)
- Small amount of text (under 1K documents)
- Domain similar to pre-training data
- Limited training time

### Domain Adaptation Pipeline

Complete pipeline for building domain expert:

1. Continued Pre-training (this notebook)
   - Train on 10K-1M domain documents
   - 1000-5000 steps
   - Model learns domain facts

2. Supervised Fine-tuning (Colab 1 or 2)
   - Train on domain-specific instruction pairs
   - 100-500 steps
   - Model learns domain tasks

3. DPO Alignment (Colab 3, optional)
   - Train on domain-specific preferences
   - 50-200 steps
   - Model aligns to domain standards

Result: Domain expert that knows facts and follows instructions!

### Data Sources by Domain

Medical:
- PubMed Central articles
- Medical textbooks
- Clinical guidelines
- MIMIC-III (with proper access)

Legal:
- Legal case databases
- Statutes and regulations
- Legal textbooks
- Pile of Law dataset

Technical:
- API documentation
- Technical blogs
- GitHub repositories
- Stack Overflow Q&A

Multilingual:
- OSCAR corpus
- CC-100
- Wikipedia in target language
- Native language books/news

### Troubleshooting

Issue: Model not retaining knowledge
- Train for more steps (500-1000+)
- Increase learning rate (5e-4)
- More diverse domain text needed
- Check data quality

Issue: Catastrophic forgetting
- Mix in 20% general data
- Lower learning rate
- Shorter training duration
- Use LoRA to preserve base model

Issue: Model outputs training data verbatim
- Overfitting - reduce epochs
- More diverse training data
- Add regularization (dropout, weight decay)

### Best Practices

1. Start with high-quality, cleaned text
2. Deduplicate to avoid memorization
3. Mix 80% domain + 20% general data
4. Monitor perplexity on domain text
5. Test knowledge retention regularly
6. Follow with instruction fine-tuning
7. Evaluate on domain benchmarks

### Evaluation Methods

Perplexity:
```python
# Lower perplexity = better domain understanding
domain_ppl = evaluate_perplexity(model, domain_test_set)
general_ppl = evaluate_perplexity(model, general_test_set)
```

Factual QA:
```python
# Test fact recall
questions = [
    "What is [domain concept]?",
    "Who invented [domain term]?",
    "When was [domain event]?",
]

for q in questions:
    answer = model.generate(q)
    score = evaluate_accuracy(answer, ground_truth)
```

Domain Benchmarks:
- Medical: MedQA, PubMedQA
- Legal: LegalBench
- Code: HumanEval for specific languages
- General: MMLU (to check forgetting)

### Extensions

#### Curriculum Learning

```python
# Start general, move to specific
stage1_data = broad_domain_text
stage2_data = narrow_specialty_text
stage3_data = highly_specific_text

train(model, stage1_data, epochs=3)
train(model, stage2_data, epochs=2)
train(model, stage3_data, epochs=1)
```

#### Vocabulary Extension

Add domain-specific tokens:
```python
# Add new tokens to tokenizer
new_tokens = ["TechCorp", "CloudAI", "DataFlow"]
tokenizer.add_tokens(new_tokens)

# Resize model embeddings
model.resize_token_embeddings(len(tokenizer))

# Then continue pre-training
```

### Resources

- Unsloth Pre-training Guide: https://docs.unsloth.ai/basics/continued-pretraining
- Domain Corpora: OSCAR, CC-100, Pile of Law
