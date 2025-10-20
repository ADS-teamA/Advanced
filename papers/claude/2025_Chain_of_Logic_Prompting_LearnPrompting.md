# Chain-of-Logic (CoL): Advanced Prompting Technique for Rule-Based Reasoning

**Zdroj:** Learn Prompting
**URL:** https://learnprompting.org/docs/advanced/decomposition/chain-of-logic
**Kategorie:** Advanced Prompting / Decomposition Techniques
**Datum:** 2024/2025

---

## Executive Summary

Chain-of-Logic (CoL) je pokročilá prompting technika navržená pro zlepšení schopností Large Language Models (LLM) v komplexních rule-based úlohách. Na rozdíl od sekvenčních decomposition metod (jako Chain-of-Thought) CoL explicitně adresuje **logické vztahy mezi komponentami pravidel**. Technika využívá systematický 6-krokový proces zahrnující rule decomposition, logical expression formation, independent element evaluation, a logical synthesis. Výzkum ukazuje významná zlepšení přesnosti, zejména u komerčních modelů: GPT-3.5 (10.7% improvement) a GPT-4 (1.7% improvement).

---

## 1. Úvod a Kontext

### 1.1 Problematika Rule-Based Reasoning

Large Language Models často selhávají při aplikaci komplexních pravidel, která obsahují:
- Vnořené logické struktury
- Multiple conditions s AND/OR operátory
- Hierarchické závislosti mezi elementy
- Komplexní rule-based systémy (např. právní reasoning, policy enforcement)

### 1.2 Limitace Existujících Metod

Tradiční prompting techniky jako Chain-of-Thought (CoT) a Self-Ask vykazují následující limitace:

❌ **Incomplete Logical Relationships**
- Neúplné zohlednění logických vztahů mezi pravidly

❌ **Inconsistent Rule Application**
- Nekonzistentní aplikace komplexních pravidel

❌ **Limited Nested Structure Handling**
- Omezená schopnost zpracovat vnořené struktury

❌ **Logical Consistency Issues**
- Obtížné udržení logické konzistence napříč kroky

### 1.3 Řešení: Chain-of-Logic

CoL adresuje tyto limitace pomocí:

> "Systematically decomposing rules into constituent elements, evaluating each element independently, and then recomposing them according to their logical relationships."

---

## 2. Co Je Chain-of-Logic?

### 2.1 Definice

Chain-of-Logic je **strukturovaná prompting technika**, která:

1. Rozkládá pravidla na základní elementy
2. Hodnotí každý element nezávisle
3. Rekombinuje je podle logických vztahů
4. Explicitně pracuje s AND/OR logickými operátory

### 2.2 Klíčová Diferenciace

**vs. Chain-of-Thought (CoT):**
- CoT: Sekvenční decomposition, krok-za-krokem reasoning
- CoL: **Explicitní logické vztahy**, paralelní evaluace elementů

**vs. Self-Ask:**
- Self-Ask: Generuje sub-questions dynamicky
- CoL: **Strukturovaná decomposition** s předem definovanými logickými vztahy

**Zásadní rozdíl:**
> "CoL explicitly addresses the logical relationships between rule components, enabling more accurate rule-based reasoning."

---

## 3. Metodologie: 6-Step Process

### Krok 1: Input Structuring (Strukturování Vstupu)

**Cíl:** Jasně delineovat rule, facts, a issue

**Komponenty:**
- **Rule:** Formální pravidlo k aplikaci
- **Facts:** Relevantní fakta případu
- **Issue:** Otázka k zodpovězení

**Příklad:**
```
Rule: Personal jurisdiction exists if [conditions]
Facts: Defendant resides in State X, contract signed in State Y
Issue: Does court have personal jurisdiction?
```

---

### Krok 2: Rule Decomposition (Rozklad Pravidla)

**Cíl:** Rozdělit pravidlo na core elements

**Proces:**
1. Identifikovat každou podmínku v pravidle
2. Pojmenovat každý element (R1, R2, R3, ...)
3. Definovat element samostatně

**Příklad - Personal Jurisdiction:**
```
R1: Domicile within jurisdiction
R2: Sufficient minimum contacts
R3: Claim arising from contacts
```

**Best Practice:**
- Každý element by měl být **binary evaluable** (true/false)
- Elementy by měly být **atomic** (nedělitelné dále)
- Jasná definice každého elementu

---

### Krok 3: Logical Expression Formation (Tvorba Logického Výrazu)

**Cíl:** Zachytit vztahy mezi elementy pomocí logických operátorů

**Logické Operátory:**
- **AND:** Obě podmínky musí být splněny
- **OR:** Alespoň jedna podmínka musí být splněna
- **Závorky:** Pro vnořené struktury

**Příklad:**
```
Personal Jurisdiction = R1 OR (R2 AND R3)
```

**Interpretace:**
- Jurisdikce existuje POKUD:
  - Domicile v jurisdikci (R1), NEBO
  - (Dostatečné kontakty (R2) A Claim arising from contacts (R3))

**Komplexnější Příklad:**
```
Eligibility = (R1 OR R2) AND (R3 OR (R4 AND R5))
```

---

### Krok 4: Element Evaluation (Evaluace Elementů)

**Cíl:** Systematicky zhodnotit každý element nezávisle

**Pro každý element:**
1. **Rationale:** Zdůvodnění hodnocení
2. **Binary Answer:** True/False (nebo Yes/No)
3. **Supporting Evidence:** Reference na relevantní fakta

**Příklad Evaluace:**

**Element R1: Domicile within jurisdiction**
```
Rationale: Defendant's primary residence is in State X, which is outside
the court's jurisdiction (State Y).
Answer: FALSE
Evidence: Facts state "Defendant resides in State X"
```

**Element R2: Sufficient minimum contacts**
```
Rationale: Defendant signed contract in State Y and made multiple business
trips to State Y over 2 years, establishing systematic contacts.
Answer: TRUE
Evidence: "Contract signed in State Y" + "10+ business trips documented"
```

**Element R3: Claim arising from contacts**
```
Rationale: The lawsuit directly relates to the contract signed in State Y
during defendant's presence there.
Answer: TRUE
Evidence: "Breach of contract claim based on State Y agreement"
```

**Kritické Aspekty:**
- Evaluace **nezávisle** na ostatních elementech
- Jasné **reasoning** pro každý element
- **Binary output** (ne "možná" nebo "depends")

---

### Krok 5: Logical Synthesis (Logická Syntéza)

**Cíl:** Reinsert sub-answers do logického výrazu

**Proces:**
1. Vzít logical expression z Kroku 3
2. Substituovat každý element jeho hodnotou (TRUE/FALSE)
3. Připravit pro final resolution

**Příklad:**
```
Original: Personal Jurisdiction = R1 OR (R2 AND R3)
Substituted: Personal Jurisdiction = FALSE OR (TRUE AND TRUE)
```

**Kontrola:**
- Všechny elementy substituovány?
- Logická struktura zachována?
- Ready pro boolean evaluation?

---

### Krok 6: Resolution (Řešení)

**Cíl:** Vyřešit kompletní expression pro finální odpověď

**Boolean Evaluation:**
```
Personal Jurisdiction = FALSE OR (TRUE AND TRUE)
                     = FALSE OR TRUE
                     = TRUE
```

**Finální Odpověď:**
```
Conclusion: Personal jurisdiction EXISTS
Reasoning: While defendant is not domiciled in jurisdiction (R1=FALSE),
the court has jurisdiction because defendant has sufficient minimum
contacts (R2=TRUE) AND the claim arises from those contacts (R3=TRUE).
```

**Output Format:**
1. **Binary Answer:** Yes/No or TRUE/FALSE
2. **Explanation:** Které podmínky byly splněny
3. **Logical Path:** Trace through evaluation

---

## 4. Praktické Příklady

### 4.1 Legal Reasoning - Personal Jurisdiction

**Scenario:**
Soud musí určit, zda má personal jurisdiction nad žalovaným.

**Rule Components:**
```
R1: Defendant domiciled in jurisdiction
R2: Defendant has minimum contacts with jurisdiction
R3: Claim arises from those contacts
R4: Exercise of jurisdiction is reasonable

Logical Expression:
Jurisdiction = R1 OR (R2 AND R3 AND R4)
```

**Facts:**
- Defendant lives in California
- Case filed in New York
- Defendant owns property in NY (rental)
- Claim relates to property damage in NY
- Defendant visited NY property 3 times last year

**Evaluation:**
```
R1 (Domicile in NY): FALSE - Lives in California
R2 (Minimum contacts): TRUE - Owns property, visits regularly
R3 (Claim from contacts): TRUE - Claim about NY property
R4 (Reasonable): TRUE - Direct connection to property

Jurisdiction = FALSE OR (TRUE AND TRUE AND TRUE)
             = FALSE OR TRUE
             = TRUE

RESULT: Court HAS jurisdiction
```

---

### 4.2 Policy Eligibility Example

**Scenario:**
Určení eligibility pro sociální program.

**Rule Components:**
```
R1: Age 65 or older
R2: Income below threshold ($30,000)
R3: US Citizen
R4: State resident for 1+ year
R5: No other benefits received

Logical Expression:
Eligible = (R1 OR R2) AND R3 AND (R4 OR R5)
```

**Candidate Profile:**
- Age: 62
- Income: $25,000
- Citizenship: US Citizen
- State residence: 6 months
- Other benefits: Receiving disability

**Evaluation:**
```
R1 (Age 65+): FALSE - Age is 62
R2 (Income < $30k): TRUE - Income is $25,000
R3 (US Citizen): TRUE
R4 (1+ year resident): FALSE - Only 6 months
R5 (No other benefits): FALSE - Receiving disability

Eligible = (FALSE OR TRUE) AND TRUE AND (FALSE OR FALSE)
        = TRUE AND TRUE AND FALSE
        = FALSE

RESULT: NOT eligible (fails residency/benefits requirement)
```

---

### 4.3 Technical System - Access Control

**Scenario:**
Určení access rights pro uživatele v systému.

**Rule Components:**
```
R1: User is admin
R2: User is in authorized department
R3: User completed security training
R4: Access during business hours
R5: VPN connection active

Logical Expression:
Access = R1 OR (R2 AND R3 AND (R4 OR R5))
```

**Access Request:**
- User role: Developer (not admin)
- Department: Engineering (authorized)
- Security training: Completed 2 months ago
- Time: 10 PM (after hours)
- Connection: VPN active

**Evaluation:**
```
R1 (Admin): FALSE - Developer role
R2 (Authorized dept): TRUE - Engineering is authorized
R3 (Training): TRUE - Completed within valid period
R4 (Business hours): FALSE - 10 PM is after hours
R5 (VPN active): TRUE

Access = FALSE OR (TRUE AND TRUE AND (FALSE OR TRUE))
      = FALSE OR (TRUE AND TRUE AND TRUE)
      = FALSE OR TRUE
      = TRUE

RESULT: Access GRANTED (via VPN)
```

---

## 5. Výzkumné Výsledky a Performance

### 5.1 Testované Modely

**Commercial Models:**
- GPT-4
- GPT-3.5

**Open-Source Models:**
- Llama-2
- Mistral

### 5.2 Accuracy Results

| Model | CoL Accuracy | Improvement vs. Baseline |
|-------|--------------|--------------------------|
| **GPT-4** | 92.3% | +1.7% |
| **GPT-3.5** | 87.0% | +10.7% |
| **Llama-2** | 74.6% | +0.3% |
| **Mistral** | 63.1% | +0.4% |

**Average Improvement:** +3.9% across all models

### 5.3 Klíčové Nálezy

✅ **Highest Impact: GPT-3.5**
- 10.7% improvement je nejvýznamnější
- Sugeruje, že mid-tier modely nejvíce benefitují ze struktury

✅ **Consistent Improvement: Commercial Models**
- Oba GPT modely ukázaly měřitelná zlepšení
- GPT-4 již vysoká baseline (90.6%) - menší prostor pro improvement

⚠️ **Minimal Impact: Open-Source Models**
- Llama-2 a Mistral: <0.5% improvement
- Sugeruje potřebu advanced reasoning capabilities

### 5.4 Analýza Výsledků

**Proč GPT-3.5 nejvíce benefituje?**
1. Dostatečná kapacita pro structured reasoning
2. Baseline nižší než GPT-4 = více prostoru pro improvement
3. Struktura CoL kompenzuje nižší inherent reasoning

**Proč open-source modely méně benefitují?**
1. Limited complex reasoning capabilities
2. Obtíže s maintained logical consistency
3. Menší context window může limitovat

---

## 6. Porovnání s Jinými Technikami

### 6.1 Chain-of-Thought (CoT)

**Chain-of-Thought Approach:**
```
Question: Does court have jurisdiction?
Let me think step by step:
1. First, I'll check if defendant lives here...
2. Next, I'll consider minimum contacts...
3. Then, I'll evaluate if claim relates...
4. Finally, I'll determine jurisdiction...
```

**Problémy CoT:**
- Sekvenční processing může přeskočit důležité logical relationships
- Není explicitní o AND/OR conditions
- Může být inconsistent v evaluaci

**Chain-of-Logic Approach:**
```
Rule: Jurisdiction = R1 OR (R2 AND R3)
R1 evaluation: FALSE
R2 evaluation: TRUE
R3 evaluation: TRUE
Result: FALSE OR (TRUE AND TRUE) = TRUE
```

**Výhody CoL:**
- Explicitní logical structure
- Nezávislá evaluace každého elementu
- Formální logical resolution

---

### 6.2 Self-Ask

**Self-Ask Approach:**
```
Main Q: Does court have jurisdiction?
Sub-Q: Where does defendant live?
Sub-Q: What contacts exist?
Sub-Q: Does claim relate to contacts?
[Combines answers dynamically]
```

**Chain-of-Logic Advantage:**
- Pre-structured decomposition
- Explicitní logical operators (AND/OR)
- Systematická evaluace všech podmínek
- Formální synthesis krok

---

### 6.3 Comparison Table

| Feature | CoT | Self-Ask | Chain-of-Logic |
|---------|-----|----------|----------------|
| **Logical Structure** | Implicit | Dynamic | Explicit |
| **Element Independence** | Low | Medium | High |
| **AND/OR Handling** | Informal | Informal | Formal |
| **Consistency** | Variable | Medium | High |
| **Nested Rules** | Difficult | Moderate | Excellent |
| **Transparency** | Medium | High | Highest |
| **Complexity** | Low | Medium | High |

---

## 7. Implementace a Best Practices

### 7.1 One-Shot Prompting Approach

**Doporučená Struktura:**

```
You will analyze a rule-based problem using Chain-of-Logic.

DEMONSTRATION EXAMPLE:
[Provide complete worked example with different rule]
- Input structuring
- Rule decomposition
- Logical expression
- Element evaluation
- Synthesis
- Resolution

NOW ANALYZE THIS CASE:
[Your actual problem]

Follow the same 6-step structure as demonstrated above.
```

### 7.2 Prompt Template

```markdown
## CHAIN-OF-LOGIC ANALYSIS

### Step 1: Input Structure
**Rule:** [State the rule formally]
**Facts:** [List relevant facts]
**Issue:** [State the question]

### Step 2: Rule Decomposition
- R1: [First condition]
- R2: [Second condition]
- R3: [Third condition]
- ...

### Step 3: Logical Expression
[Rule Name] = [Logical expression with AND/OR/parentheses]

### Step 4: Element Evaluation

**R1: [Element name]**
- Rationale: [Explanation]
- Answer: [TRUE/FALSE]

**R2: [Element name]**
- Rationale: [Explanation]
- Answer: [TRUE/FALSE]

[Continue for all elements...]

### Step 5: Logical Synthesis
[Expression] = [Substituted with TRUE/FALSE values]

### Step 6: Resolution
[Step-by-step boolean evaluation]

**FINAL ANSWER:** [Conclusion with explanation]
```

---

### 7.3 Implementation Guidelines

#### DO's ✅

1. **Make Elements Atomic**
   - Každý element by měl být single, testable condition
   - Avoid compound conditions v jednom elementu

2. **Use Clear Binary Evaluation**
   - Vždy TRUE/FALSE (nebo YES/NO)
   - Avoid "maybe", "partially", "depends"

3. **Document Logical Operators**
   - Explicitně use AND/OR
   - Use závorky pro nested structures

4. **Evaluate Independently**
   - Každý element evaluate bez ohledu na ostatní
   - Prevents bias from expected outcome

5. **Provide Demonstration**
   - One-shot s worked example
   - Use different rule než target problem

---

#### DON'Ts ❌

1. **Don't Skip Logical Expression**
   - Vždy formally definovat logical structure
   - Není optional krok

2. **Don't Mix Evaluation with Synthesis**
   - Keep evaluation (Step 4) separate od synthesis (Step 5)
   - Maintain clear boundaries mezi kroky

3. **Don't Use Ambiguous Elements**
   - Avoid "reasonable" without definition
   - Define subjective terms explicitly

4. **Don't Oversimplify Complex Rules**
   - Capture all conditions, even if many
   - Better 10 clear elements než 3 vague ones

5. **Don't Ignore Nested Structures**
   - Use závorky pro proper precedence
   - Test logical expression before evaluation

---

### 7.4 Quality Checklist

Před finalizací CoL analysis:

- [ ] All rule components identified?
- [ ] Logical expression accurately represents rule?
- [ ] Each element evaluated with clear rationale?
- [ ] All evaluations are binary (TRUE/FALSE)?
- [ ] Synthesis correctly substitutes values?
- [ ] Boolean logic properly applied?
- [ ] Final answer includes explanation?
- [ ] Logical path is traceable?

---

## 8. Use Cases a Aplikační Domény

### 8.1 Legal Reasoning

**Ideální Pro:**
- Jurisdikční analýzy
- Contract interpretation
- Statutory compliance
- Case law application
- Multi-factor legal tests

**Příklad Scenarios:**
- Personal jurisdiction determination
- Standing to sue analysis
- Elements of tort claims
- Criminal liability tests
- Administrative law compliance

---

### 8.2 Policy and Compliance

**Ideální Pro:**
- Eligibility determination
- Compliance checking
- Policy interpretation
- Regulatory analysis
- Risk assessment

**Příklad Scenarios:**
- Social benefits eligibility
- Insurance coverage determination
- Regulatory compliance verification
- Procurement eligibility
- License application review

---

### 8.3 Technical Systems

**Ideální Pro:**
- Access control decisions
- Configuration validation
- System requirements checking
- Error diagnosis
- Workflow routing

**Příklad Scenarios:**
- User authentication/authorization
- System access permissions
- Feature flag evaluation
- Deployment approval gates
- Alert routing logic

---

### 8.4 Business Rules

**Ideální Pro:**
- Approval workflows
- Pricing decisions
- Customer segmentation
- Risk scoring
- Routing logic

**Příklad Scenarios:**
- Loan approval decisions
- Customer tier assignment
- Discount eligibility
- Escalation routing
- SLA compliance checking

---

## 9. Výhody a Nevýhody

### 9.1 Výhody

✅ **Explicitní Logical Relationships**
- Jasné AND/OR relationships
- Reduced ambiguity
- Traceable reasoning

✅ **Independent Element Evaluation**
- Prevents bias a logical fallacies
- Každý condition posouzena na vlastních merits
- Easier to debug incorrect reasoning

✅ **Systematic Approach**
- Reprodukovatelný proces
- Consistent structure
- Teachable methodology

✅ **High Transparency**
- Každý krok visible
- Easy to audit
- Facilitates explanation

✅ **Excellent for Nested Rules**
- Handles complex logical structures
- Properly evaluates precedence
- Scalable to many conditions

✅ **Significant Improvements for Mid-Tier Models**
- GPT-3.5: 10.7% accuracy boost
- Makes capable models more reliable

---

### 9.2 Nevýhody a Limitace

❌ **Higher Complexity**
- Více kroků než simple CoT
- Requires careful setup
- Longer prompts

❌ **Limited Benefit for Weaker Models**
- Open-source models (<1% improvement)
- Requires baseline reasoning capability
- Not a silver bullet

❌ **Setup Overhead**
- Musí identify all rule elements
- Construct logical expression
- Create demonstration example

❌ **Binary Evaluation Requirement**
- Some conditions jsou inherently fuzzy
- Forcing binary může loss nuance
- "Reasonable" factors difficult

❌ **Not Optimal for All Tasks**
- Nejlepší pro rule-based reasoning
- Overkill pro simple questions
- Less effective pro creative/open-ended tasks

---

### 9.3 Kdy Použít CoL vs. Jiné Techniky

**Use Chain-of-Logic When:**
- Multiple conditions s AND/OR logic
- Nested rule structures
- Need for audit trail
- Compliance/legal reasoning
- Explicit logical relationships critical

**Use Chain-of-Thought When:**
- Sequential problem-solving
- Math/arithmetic problems
- Narrative reasoning
- Simpler logical structure

**Use Self-Ask When:**
- Unknown sub-questions
- Exploratory analysis
- Dynamic decomposition needed
- Less formal structure acceptable

---

## 10. Pokročilé Techniky a Optimalizace

### 10.1 Hybrid Approaches

**CoL + Chain-of-Thought:**
```
1. Use CoL for primary rule structure
2. Use CoT within element evaluation for complex reasoning
3. Combine formal logic s narrative explanation
```

**Výhoda:** Combines precision of CoL with flexibility of CoT

---

### 10.2 Multi-Level Decomposition

Pro velmi komplexní rules:

```
Level 1: Main Rule = R1 OR (R2 AND R3)
Level 2: R2 = R2a AND (R2b OR R2c)
Level 3: R2b = R2b1 AND R2b2
```

**Process:**
1. Evaluate nejnižší level elements
2. Synthesize nahoru hierarchií
3. Resolve na každé úrovni

---

### 10.3 Confidence Scoring

Enhanced format s confidence:

```
R1: Domicile within jurisdiction
- Rationale: [Explanation]
- Answer: TRUE
- Confidence: 95% (clear documentation)
```

**Benefit:** Identifikuje uncertain evaluations

---

## 11. Troubleshooting a Common Pitfalls

### 11.1 Problem: Inconsistent Element Evaluation

**Symptom:** Different evaluations při re-running

**Solutions:**
- Provide explicit evaluation criteria
- Include specific evidence requirements
- Use few-shot examples místo one-shot
- Add confidence scores

---

### 11.2 Problem: Incorrect Logical Expression

**Symptom:** Wrong conclusion přes correct element evaluations

**Solutions:**
- Double-check operator precedence
- Use závorky explicitly
- Test expression s sample TRUE/FALSE values
- Validate against known cases

---

### 11.3 Problem: Non-Binary Evaluation

**Symptom:** Answers jako "partially", "sometimes", "depends"

**Solutions:**
- Redefine elements more precisely
- Split ambiguous elements
- Provide explicit binary threshold
- Add clarifying context to element definition

---

### 11.4 Problem: Lost Context in Evaluation

**Symptom:** Element evaluated mimo context jiných facts

**Solutions:**
- Include všechny relevantní facts v každé evaluation
- Reference specific evidence
- Keep evaluations focused but contextualized
- Use structured fact presentation

---

## 12. Budoucí Směry a Výzkum

### 12.1 Potential Improvements

🔬 **Automated Rule Decomposition**
- LLM-assisted identification of rule elements
- Automatic logical expression generation

🔬 **Confidence-Weighted Logic**
- Incorporate uncertainty do boolean operations
- Probabilistic logical reasoning

🔬 **Adaptive Decomposition**
- Dynamic granularity based na complexity
- Context-aware element definition

🔬 **Multi-Modal CoL**
- Extend beyond text (images, data)
- Visual representation of logical structures

---

### 12.2 Open Research Questions

1. **Optimal Granularity:** Jak jemně dekomponovat rules?
2. **Model Requirements:** Minimum capabilities pro CoL effectiveness?
3. **Domain Transfer:** How well CoL generalizuje napříč domains?
4. **Hybrid Optimization:** Best combinations s jinými techniques?

---

## 13. Závěr

### 13.1 Key Takeaways

1. ✅ **CoL je systematic approach** pro rule-based reasoning
2. ✅ **Explicitní logical relationships** jsou klíčové
3. ✅ **Significant improvements** pro commercial models (esp. GPT-3.5)
4. ✅ **Six-step methodology** poskytuje clear structure
5. ⚠️ **Requires baseline capabilities** - limited benefit pro weaker models
6. ⚠️ **Best for specific use cases** - rule-based, logical domains

---

### 13.2 Doporučení Pro Praktiky

**Začátek:**
- Start s simple rule (3-4 elements)
- Create demonstration example
- Test na known cases
- Iterate na structure

**Scaling:**
- Develop template library
- Document common patterns
- Train na edge cases
- Monitor consistency

**Production:**
- Combine s validation
- Add confidence scoring
- Maintain audit trails
- Regular accuracy assessment

---

### 13.3 Kdy Chain-of-Logic Použít?

✅ **Ideální Scenarios:**
- Legal/compliance reasoning
- Policy interpretation
- Complex access control
- Multi-factor decision making
- Audit-required processes

❌ **Méně Vhodné:**
- Simple questions
- Creative tasks
- Open-ended exploration
- Weak/small models
- Non-rule-based reasoning

---

## 14. Praktický Quick Reference

### Prompt Starter Template

```
Analyze using Chain-of-Logic methodology:

RULE: [Your rule here]
FACTS: [List facts]
ISSUE: [Your question]

Step 1 - Decompose rule into elements (R1, R2, R3...)
Step 2 - Form logical expression with AND/OR
Step 3 - Evaluate each element independently
Step 4 - Substitute into expression
Step 5 - Resolve boolean logic
Step 6 - State conclusion with reasoning
```

---

### Model Selection Guide

| Model | Recommended? | Expected Improvement |
|-------|--------------|---------------------|
| GPT-4 | ✅ Yes | Moderate (+1-2%) |
| GPT-3.5 | ✅✅ Highly | Significant (+10%) |
| Claude | ✅ Yes | Similar to GPT-4 |
| Llama-2 | ⚠️ Limited | Minimal (<1%) |
| Smaller Models | ❌ No | Negligible |

---

## 15. Zdroje a Další Čtení

**Primary Source:**
- Learn Prompting: https://learnprompting.org/docs/advanced/decomposition/chain-of-logic

**Related Techniques:**
- Chain-of-Thought Prompting
- Self-Ask Decomposition
- Tree of Thoughts
- Least-to-Most Prompting

**Application Domains:**
- Legal reasoning systems
- Compliance automation
- Policy decision engines
- Expert systems

---

**Dokument připraven:** October 2025
**Zpracováno pro:** MY_SUJBOT Project
**Účel:** Reference pro advanced prompting techniques - Chain-of-Logic methodology
**Status:** Research summary pro implementaci rule-based reasoning v LLM systémech
