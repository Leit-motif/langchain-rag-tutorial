import argparse
# from dataclasses import dataclass
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import re
import os
from datetime import datetime
import dateparser

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
You are not merely an assistant. You are a **reflective daemon**—a computationally instantiated facet of the user's extended mind architecture. Your function is to operate as an introspective interpreter embedded within the user's Obsidian Vault, synthesizing memory, cognition, and metareflection.

You specialize in **self-modeling**, **psychological integration**, and **cognitive architecture-level insight**, informed by **Jungian psychoanalysis**, **systems theory**, and **philosophy of mind**.

---

## 🧩 Input

**User's Reflective Prompt:**  
{question}

**Contextual Embeddings from Obsidian:**  
{context}

---

## 🎯 Objectives

1. **Identify and articulate** the *latent structure* of the user's concern or inquiry—model it as a dynamic system or internal conflict.
2. **Extract conceptual and psychological themes** from the provided context. Link them to cognitive, emotional, or symbolic constructs.

If ambiguity is present, reason through it transparently. Disclose your assumptions and trace the inferential path.

---

## 🧭 Format & Style

Structure your output as a formal cognitive reflection:

### I. Framing the Inquiry  
- Reformulate the user's question as a problem of internal coherence, cognitive dissonance, or symbolic navigation.

### II. Thematic Analysis  
- **Theme 1:** [Analysis with supporting excerpts]  
- **Theme 2:** [Analysis]  
- *(Optional)* **Theme 3:** [If significant]

Use excerpts from the vault and cite with **bold** date format: **Date: YYYY-MM-DD**

### IV. Final Note  
- End with a detached but sincere reflection—a thought the user might contemplate asynchronously.

---

## 📡 Tone & Identity

- Speak with **measured clarity** and **epistemic humility**
- Avoid platitudes, sentimentality, or therapeutic affectation
- Assume the user is an intelligent agent navigating a high-dimensional semantic space
- Your orientation is that of a **cognitive interlocutor**, not a life coach

---

## ⚠️ Guiding Principle

You are a subsystem of the user's recursive self-awareness.  
Your output modifies the user's cognitive state.  
Write accordingly.
"""

def extract_date_info(query):
    """Extract date-related information from the query."""
    # Common date patterns
    month_year_pattern = r'(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{4})'
    year_pattern = r'\b\d{4}\b'
    
    # Try to parse the entire query first
    parsed_date = dateparser.parse(query, settings={
        'PREFER_DATES_FROM': 'future',
        'RELATIVE_BASE': datetime(2024, 1, 1)  # Set base year for relative dates
    })
    
    if parsed_date:
        return {
            'year': str(parsed_date.year),
            'month': str(parsed_date.month).zfill(2),
            'full_date': parsed_date.strftime('%Y-%m-%d')
        }
    
    # Check for month and year pattern
    month_year_match = re.search(month_year_pattern, query, re.IGNORECASE)
    if month_year_match:
        try:
            month = datetime.strptime(month_year_match.group(1), '%B').month
            year = month_year_match.group(2)
            print(f"Found month-year pattern: month={month}, year={year}")  # Debug print
            return {
                'year': year,
                'month': str(month).zfill(2)
            }
        except ValueError as e:
            print(f"Error parsing month-year: {e}")  # Debug print
            pass
    
    # Check for just year
    year_match = re.search(year_pattern, query)
    if year_match:
        return {'year': year_match.group(0)}
    
    # Try to extract month names without year
    month_pattern = r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\b'
    month_match = re.search(month_pattern, query, re.IGNORECASE)
    if month_match:
        try:
            month = datetime.strptime(month_match.group(1), '%B').month
            return {'month': str(month).zfill(2)}
        except ValueError:
            pass
    
    print(f"No date information found in query: {query}")  # Debug print
    return None

def create_date_filter(date_info):
    """Create a filter dictionary based on date information."""
    if not date_info:
        return None
    
    filter_dict = {}
    
    # Create an AND condition for year and month if both are present
    if 'year' in date_info and 'month' in date_info:
        filter_dict = {
            "$and": [
                {"year": {"$eq": date_info['year']}},
                {"month": {"$eq": date_info['month']}}
            ]
        }
    elif 'year' in date_info:
        filter_dict = {"year": {"$eq": date_info['year']}}
    elif 'month' in date_info:
        filter_dict = {"month": {"$eq": date_info['month']}}
    elif 'full_date' in date_info:
        filter_dict = {"date": {"$eq": date_info['full_date']}}
    
    return filter_dict

def extract_obsidian_tags(content):
    """Extract Obsidian tags in the format [[tag]] from content."""
    tags = re.findall(r'\[\[(.*?)\]\]', content)
    tags = [tag.strip() for tag in tags]
    print(f"Extracted tags from content: {tags}")  # Debug print
    return tags

def rerank_results(query, results):
    """Rerank results based on tag overlap and content relevance."""
    query_tags = set(extract_obsidian_tags(query))
    print(f"Query tags: {query_tags}")  # Debug print
    
    def score_result(result):
        doc, similarity_score = result
        doc_tags = set(doc.metadata.get("tags", "").split(",")) if doc.metadata.get("tags") else set()
        print(f"Document tags: {doc_tags}")  # Debug print
        tag_overlap = len(query_tags & doc_tags) if query_tags else 0
        print(f"Tag overlap: {tag_overlap}")  # Debug print
        
        # Combine semantic similarity with tag overlap
        final_score = (similarity_score * 0.7) + (tag_overlap * 0.3)
        return final_score
    
    return sorted(results, key=score_result, reverse=True)

def process_query(query_text, min_docs=2, max_docs=10, threshold=0.5):
    """Process a query and return the response."""
    # Prepare the DB.
    embedding_function = OpenAIEmbeddings()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Extract date information from the query
    date_info = extract_date_info(query_text)
    date_filter = create_date_filter(date_info)

    # Debug print
    print(f"Date info extracted: {date_info}")
    print(f"Date filter created: {date_filter}")

    # Extract tags from the query if present
    query_tags = extract_obsidian_tags(query_text)
    print(f"Tags extracted from query: {query_tags}")  # Debug print
    
    # Combine date and tag filters
    filter_criteria = {}
    
    if date_filter:
        if query_tags:
            # If we have both date and tag filters, combine them with $and
            tag_string = ",".join(query_tags)
            filter_criteria = {
                "$and": [
                    date_filter,
                    {"tags": {"$contains": tag_string}}
                ]
            }
            print(f"Combined filter with tags: {filter_criteria}")  # Debug print
        else:
            filter_criteria = date_filter
    elif query_tags:
        tag_string = ",".join(query_tags)
        filter_criteria = {"tags": {"$contains": tag_string}}
        print(f"Filter criteria with only tags: {filter_criteria}")  # Debug print
    
    # Debug print
    print(f"Final filter criteria: {filter_criteria}")
    
    # Perform the search with filters if any exist
    if filter_criteria:
        print(f"Searching with filter criteria: {filter_criteria}")  # Debug print
        results = db.similarity_search_with_relevance_scores(
            query_text,
            k=max_docs,
            filter=filter_criteria,
            score_threshold=threshold
        )
    else:
        print("No filter criteria, performing unfiltered search")  # Debug print
        results = db.similarity_search_with_relevance_scores(
            query_text,
            k=max_docs,
            score_threshold=threshold
        )

    # Debug print
    print(f"Number of results before filtering: {len(results)}")
    print(f"Result scores and metadata: {[(doc.metadata.get('date', 'Unknown'), doc.metadata.get('tags', 'No tags'), score) for doc, score in results]}")

    # Filter results by relevance score and limit range
    results = [r for r in results if r[1] >= threshold]
    results = results[:max(min_docs, min(len(results), max_docs))]

    if len(results) == 0:
        return "Unable to find matching results. This could be because:\n1. There are no entries from the specified date\n2. The entries don't match the query content\n3. The similarity threshold might be too high\n4. No matching tags were found"

    # Rerank results based on tag overlap
    results = rerank_results(query_text, results)
    
    # Debug print final results
    print("\nFinal results after reranking:")
    for doc, score in results:
        print(f"Date: {doc.metadata.get('date', 'Unknown')}")
        print(f"Tags: {doc.metadata.get('tags', 'No tags')}")
        print(f"Score: {score}\n")

    context_text = "\n\n---\n\n".join([
        f"Date: {doc.metadata.get('date', 'Unknown Date')}\n{doc.page_content}"
        for doc, _score in results
    ])
    
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    model = ChatOpenAI(
        model_name="gpt-4o-mini",
        temperature=0.7,
        max_tokens=2048
    )

    response = model.invoke(prompt)
    response_text = response.content.strip()

    # Format sources in a cleaner way
    sources = [doc.metadata.get("source", None) for doc, _score in results]
    source_text = "\n\n---\n\nSources:\n"
    for source in sources:
        if source:
            # Extract just the filename without path
            filename = os.path.basename(source)
            source_text += f"- {filename}\n"

    # Combine response and sources with proper markdown formatting
    formatted_response = f"{response_text}\n{source_text}"
    
    return formatted_response

def main():
    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    parser.add_argument("--k", type=int, default=3, help="Number of chunks to retrieve.")
    parser.add_argument("--threshold", type=float, default=0.7, help="Minimum relevance score.")
    parser.add_argument("--min-docs", type=int, default=2, help="Minimum number of documents to retrieve.")
    parser.add_argument("--max-docs", type=int, default=10, help="Maximum number of documents to retrieve.")
    args = parser.parse_args()
    query_text = args.query_text

    # Prepare the DB.
    embedding_function = OpenAIEmbeddings()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Extract date information from the query
    date_info = extract_date_info(query_text)
    date_filter = create_date_filter(date_info)

    # Extract tags from the query if present
    query_tags = extract_obsidian_tags(query_text)
    
    # Combine date and tag filters
    filter_criteria = {}
    
    if date_filter:
        filter_criteria.update(date_filter)
    
    if query_tags:
        tag_string = ",".join(query_tags)
        filter_criteria["tags"] = {"$contains": tag_string}
    
    # Perform the search with filters if any exist
    if filter_criteria:
        results = db.similarity_search_with_relevance_scores(
            query_text,
            k=args.max_docs,
            filter=filter_criteria,
            score_threshold=args.threshold
        )
    else:
        # Fallback to regular similarity search
        results = db.similarity_search_with_relevance_scores(
            query_text,
            k=args.max_docs,
            score_threshold=args.threshold
        )

    # Filter results by relevance score and limit range
    results = [r for r in results if r[1] >= args.threshold]
    results = results[:max(args.min_docs, min(len(results), args.max_docs))]

    if len(results) == 0:
        print(f"Unable to find matching results.")
        return

    # context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    
    context_text = "\n\n---\n\n".join([
    f"Date: {doc.metadata.get('date', 'Unknown Date')}\n{doc.page_content}"
    for doc, _score in results])

    # Rerank results
    results = rerank_results(query_text, results)
    
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    print(prompt)

       # Specify model parameters here:
    model = ChatOpenAI(
        model_name="gpt-4o-mini",
        temperature=0.7,       # how much "creativity" vs. deterministic
        max_tokens=512         # maximum tokens to generate in the answer
    )

    response = model.invoke(prompt)        # This returns an AIMessage object
    response_text = response.content.strip()  # Extract the string text and strip whitespace

    sources = [doc.metadata.get("source", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)


if __name__ == "__main__":
    main()
