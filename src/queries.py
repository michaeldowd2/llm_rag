def SentimentStore_0():
    return [
        "{subject} market news",
        "{subject} current market sentiment analysis",
        "recent reddit posts {subject}",
        "current trading analysis {subject}",
        "{subject} latest news and trends",
        "{subject} value outlook"
    ]

def SentimentAnalysis_0():
    p = 'Is the value of {subject} likely to "Increase" or "Decrease" in the near future'
    t = """
Use only the context below to give a one word answer to the question at the end. Answer in json form with the keys "answer" and "explanation".
<context>
{context}
</context>
Q: {question}? A:
"""
    return p, t

def SentimentAnalysis_1():
    p = 'In the near future the value of {subject} is going to '
    t = """
Use only the context below to finish the statement at the end. Answer in json form with the keys "answer" and "explanation".
<context>
{context}
</context>
Statement: {question}
"""
    return p, t

def SentimentAnalysis_2():
    p = 'Market analysis shows that in the near future the value of {subject} is likely to '
    t = """
Use only the context below to finish the statement at the end. Answer in json form with the keys "answer" and "explanation".
<context>
{context}
</context>
Statement: {question}
"""
    return p, t

def SentimentAnalysis_3():
    p = 'Current outlook on the price of {subject} is '
    t = """
Use only the context below to finish the statement at the end. Answer in json form with the keys "answer" and "explanation".
<context>
{context}
</context>
Statement: {question}
"""
    return p, t

def SentimentAnalysis_4():
    p = 'In the coming weeks, the value of {subject} is expected to '
    t = """
Use only the context below to finish the statement at the end. Answer in json form with the keys "answer" and "explanation".
<context>
{context}
</context>
Statement: {question}
"""
    return p, t

def PriceStore_0():
    return [
        "Current dollar value of {subject}"
        "current price in dollars of {subject}",
        "Current Market Value of {subject}"
    ]

def FactorStore_0():
    return [
        "top {no_factors} factors that {direction} value of {subject}"
    ]

def FactorsAnalysis_0():
    p = "What are the {no_factors} biggest factors that cause the value of {subject} to {direction}"
    t = """
Answer the question at the end using the following context. Answer concisely in list form.
<context>
{context}
</context>
Q: {question}? A:
"""
    return p, t
def FactorsSplit_0():
    p = "Identify and seperate subjects in the context and create standalone bullet points. If there is only one subject, then there should only be one item in the list"
    t = """
Answer the question at the end using the following context. Answer in list form with each item just being a few words max.
<context>
{context}
</context>
Q: {question}. A:
"""

def EffectStore_0():
    return [
        "latest news on {factor} relating to {subject}",
        "Reddit update on {factor} relating to {subject}",
        "Bloomberg news on {factor} relating to {subject}"
    ]

def EffectAnalysis_0():
    p = "In the near future is {factor} likely to cause a {direction} in the value of {subject}"
    t = """
Use only the context below to give a one word answer to the question at the end. Answer in json form with the keys "answer" and "explanation".
<context>
{context}
</context>
Q: {question}? A:
"""
    return p, t

def EffectAnalysis_1():
    p = "In the near future how likely is it that {factor} will cause the value of {subject} to {direction}"
    t = """
Use only the context below to give a one word answer to the question at the end. Answer in json form with the keys "answer" and "explanation".
<context>
{context}
</context>
Q: {question}? A:
"""
    return p, t

def EffectAnalysis_2():
    p = "Is the value of {subject} likely to {direction} due to {factor} in the near future"
    t = """
Use only the context below to give a one word answer to the question at the end. Answer in json form with the keys "answer" and "explanation".
<context>
{context}
</context>
Q: {question}? A:
"""
    return p, t
