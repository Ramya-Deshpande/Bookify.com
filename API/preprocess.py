def preprocess_text(text, vectorizer, fit_vectorizer=False):
    if fit_vectorizer:
        # Fit on the input text, then transform
        vectorizer.fit([text])
    # Transform the text to vector
    return vectorizer.transform([text]).toarray()
