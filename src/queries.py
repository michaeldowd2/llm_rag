def BINGQ_Factors(subject, no_factors):
    bq = 'what causes ' + str(no_factors) + ' the price of ' + subject + ' to increase'
    return bq

def LLMQ_Factors(subject, no_factors):
    q = 'what are the' + str(no_factors) + ' single biggest factors that cause the ' + subject + ' to increase'
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
