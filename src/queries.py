def BINGQ_Factors(subject, no_factors, direction):
    bq = 'what causes ' + str(no_factors) + ' the price of ' + subject + ' to ' + direction
    return bq

def Factors_Q0():
    b = [
        "top {no_factors}  factors that cause the price of {subject} to {direction}"
    ]
    p = "what are the {no_factors} single biggest factors that cause the price of {subject} to {direction}"
    t = """
Answer the question at the end using the following context. Answer in list form.
<context>
{context}
</context>
Q: {question}? A:
"""
    return b, p, t

def Effect_Q0():
    b = [
        "latest news on {factor} relating to {subject}",
        "Reddit update on {factor} relating to {subject}",
        "Bloomberg news on {factor} relating to {subject}"
    ]
    p = "In the near future is {factor} relating to {subject} likely to occur"
    t = """
Use only the context below to give a one word answer to the question at the end. Answer in json form with the keys "answer" and "explanation".
<context>
{context}
</context>
Q: {question}? A:
"""
    return b, p, t

def Effect_Q1():
    b = [
        "latest news on {factor} relating to {subject}",
        "Reddit update on {factor} relating to {subject}",
        "Bloomberg news on {factor} relating to {subject}"
    ]
    p = "In the near future how likely is it that {factor} will cause the price of {subject} to {direction}"
    t = """
Use only the context below to give a one word answer to the question at the end. Answer in json form with the keys "answer" and "explanation".
<context>
{context}
</context>
Q: {question}? A:
"""
    return b, p, t

def Effect_Q2():
    b = [
        "latest news on {factor} relating to {subject}",
        "Reddit update on {factor} relating to {subject}",
        "Bloomberg news on {factor} relating to {subject}"
    ]
    p = "Is the price of {subject} likely to {direction} due to {factor} in the near"
    t = """
Use only the context below to give a one word answer to the question at the end. Answer in json form with the keys "answer" and "explanation".
<context>
{context}
</context>
Q: {question}? A:
"""
    return b, p, t

def LLMQ_Factors(subject, no_factors, direction):
    q = 'what are the' + str(no_factors) + ' single biggest factors that cause the ' + subject + ' to ' + direction
    t = """
Answer the question at the end using the following context. Answer in list form.
<context>
{context}
</context>
Q: {question}? A:
"""
    return q, t

def BINGQ_Effect(subject, factor):
    bq = 'How can ' + factor + ' affect the price of ' + subject
    return bq

def LLMQ_Effect(subject, factor):
    q = 'How could ' + factor + ' cause the price of ' + subject + ' to Increase'
    t = """
Give a very specific and minimal answer the question at the end using the following context.
<context>
{context}
</context>
Q: {question}? A:
"""
    return q, t
