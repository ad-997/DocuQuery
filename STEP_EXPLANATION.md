### **Steps 1 - 10**
    We've basically built an in-memory schematic engine, a question is converted to vector, every document chunk
    is also represented in vector, and you retrieve the chunks whose meanings are closest to the question (the query embedding and document chunk embedding scores are high)
    Qdrant is the next step because we don't want to regenerate and manually search all those vectors everytim