def SentimentStore_0():
    return [
        "market news {subject}"
        #"google finance news {subject}",
        #"reddit opinion on price of {subject}"
    ]

def SentimentAnalysis_0():
    p = 'Is the price of {subject} likely to "Increase" or "Decrease" in the near future'
    t = """
Use only the context below to give a one word answer to the question at the end. Answer in json form with the keys "answer" and "explanation".
<context>
{context}
</context>
Q: {question}? A:
"""
    return p, t

def SentimentAnalysis_1():
    p = 'Is the price of {subject} going "Up" or "Down"'
    t = """
Use only the context below to give a one word answer to the question at the end. Answer in json form with the keys "answer" and "explanation".
<context>
{context}
</context>
Q: {question}? A:
"""
    return p, t

def SentimentAnalysis_2():
    p = 'What do people think is happening to the price of {subject}, is it "Rising" or "Falling"'
    t = """
Use only the context below to give a one word answer to the question at the end. Answer in json form with the keys "answer" and "explanation".
<context>
{context}
</context>
Q: {question}? A:
"""
    return p, t

def PriceStore_0():
    return [
        "Current Price of {subject}"
        "Price, Volume, High and Low values for {subject}",
        "Current Market Value of {subject}"
    ]

def PriceAnalysis_0():
    p = 'What is the current price of {subject}'
    t = """
Use only the context below to give answer the question at the end. Answer in json form with the keys "answer" and "explanation".
<context>
{context}
</context>
Q: {question}? A:
"""
    return p, t

def PriceAnalysis_1():
    p = 'What price is {subject} trading at'
    t = """
Use only the context below to give answer the question at the end. Answer in json form with the keys "answer" and "explanation".
<context>
{context}
</context>
Q: {question}? A:
"""
    return p, t

def PriceAnalysis_2():
    p = 'What is the current market value of {subject}'
    t = """
Use only the context below to give answer the question at the end. Answer in json form with the keys "answer" and "explanation".
<context>
{context}
</context>
Q: {question}? A:
"""
    return p, t


def FactorsAnalysis_0():
    p = "what are the {no_factors} biggest factors that cause the price of {subject} to {direction}"
    t = """
Answer the question at the end using the following context. Answer in list form.
<context>
{context}
</context>
Q: {question}? A:
"""
    return p, t

def FactorStore_0():
    return [
        "top {no_factors} factors that cause the price of {subject} to {direction}"
    ]

def FactorsAnalysis_0():
    p = "what are the {no_factors} biggest factors that cause the price of {subject} to {direction}"
    t = """
Answer the question at the end using the following context. Answer in list form.
<context>
{context}
</context>
Q: {question}? A:
"""
    return p, t

def EffectStore_0():
    return [
        "latest news on {factor} relating to {subject}",
        "Reddit update on {factor} relating to {subject}",
        "Bloomberg news on {factor} relating to {subject}"
    ]

def EffectAnalysis_0():
    p = "In the near future is {factor} likely to cause a {direction} in the price of {subject}"
    t = """
Use only the context below to give a one word answer to the question at the end. Answer in json form with the keys "answer" and "explanation".
<context>
{context}
</context>
Q: {question}? A:
"""
    return p, t

def EffectAnalysis_1():
    p = "In the near future how likely is it that {factor} will cause the price of {subject} to {direction}"
    t = """
Use only the context below to give a one word answer to the question at the end. Answer in json form with the keys "answer" and "explanation".
<context>
{context}
</context>
Q: {question}? A:
"""
    return p, t

def EffectAnalysis_2():
    p = "Is the price of {subject} likely to {direction} due to {factor} in the near future"
    t = """
Use only the context below to give a one word answer to the question at the end. Answer in json form with the keys "answer" and "explanation".
<context>
{context}
</context>
Q: {question}? A:
"""
    return p, t

