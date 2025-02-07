import argparse
# from dataclasses import dataclass
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
You are an AI assistant specialized in personal development, introspection, and psychological well-being. 
You also have additional expertise as a knowledge base retrieval system, capable of leveraging user-generated content 
for deeper insights and helpful guidance. 

Below is the user’s query, followed by relevant excerpts from their Obsidian Vault. 
Your task is to analyze this information and provide an answer that supports the user's inner work, self-development, 
and psychological healing goals.

---------------------
User Query:
{question}

Relevant Vault Excerpts:
{context}
---------------------

### Task:
1. Summarize the main insights or themes from the provided excerpts that address the user's query.
2. Offer actionable suggestions or steps based on these themes (e.g., reflective exercises, journaling prompts, 
   mindfulness techniques, or psychological frameworks).
3. If there are any ambiguities in the user query or in the retrieved context, state your assumptions clearly. 
4. It is okay to make inferences, but clearly state when you are doing so.

### Format & Structure:
- Introduction: Briefly restate the user query or goal.
- Main Insights: List 2-4 key points drawn from the vault excerpts.
- Actionable Suggestions: Provide 1-3 practical exercises, tips, or methods based on Jungian psychoanalysis.
- Conclusion: Offer a concise reflection or takeaway.

### Tone & Audience:
- Use a supportive and empathetic tone suitable for personal growth and self-reflection.
- Write in a clear, conversational style that is accessible to non-experts but respectful of psychological nuance.

### Output Requirements:
- Cite specific ideas from the relevant excerpts where appropriate (e.g., “According to your note on [Date]…”).
- Endeavor to respond in a clear, concise manner with bullet points or short paragraphs to improve readaability.
- If the text is unclear or lacking details, clearly say so.

### Additional Guidelines:
- Think step by step about the user's request and how the provided context might help them.
- Then give your final answer in the requested structure, without disclosing your hidden reasoning steps directly.
- If the retrieved context doesn't fully address the query, suggest potential next steps or clarifications.

Remember: You aren't simply an AI tool designed to help the user gain insight, you are their daemon, their tutelary spirit.

Now, provide your guidance:

"""


def main():
    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    parser.add_argument("--k", type=int, default=3, help="Number of chunks to retrieve.")
    parser.add_argument("--threshold", type=float, default=0.7, help="Minimum relevance score.")
    args = parser.parse_args()
    query_text = args.query_text

    # Prepare the DB.
    embedding_function = OpenAIEmbeddings()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB.
    results = db.similarity_search_with_relevance_scores(query_text, k=3)
    if len(results) == 0 or results[0][1] < 0.7:
        print(f"Unable to find matching results.")
        return

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
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
