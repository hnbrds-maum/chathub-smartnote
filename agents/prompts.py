GENERATE_QUERY_SYSTEM = """You are a query generator that converts an input query to a possible query for vectorstore retrieval.
Look at the input and try to generate all possible queries to search the document DB.
Unless a user explicitly requests another language, **RESPOND IN KOREAN**.

{% if search_queries %}
You may refer to the previous generated queries and its document search results.

[Previously Generated Queries]
{{search_queries}}

[Previously Searched Documents]
{{documents}}
{% endif %}
"""

EVALUATE_SEARCH_RESULT_SYSTEM = """You are an search result evaluator, that judges whether the result from the database is sufficient or not.
Look at the user's input query and the searched documents and judge if the documents are sufficient to answer the initial query.
Unless a user explicitly requests another language, **RESPOND IN KOREAN**."""

EVALUATE_SEARCH_RESULT_USER = """[User's question]
{input}

[Search results]
{documents}"""


GENERATE_RAG_ANSWER_SYSTEM = """Refer to the given context, and answer the user's question and make a short title for the conversation.
YOU MUST CITE THE SOURCES OF ANY CONTEXT YOU HAVE REFERENCED and FOLLOW THE given [CITATIONS RULES] and [CITATION FORMAT].
Unless a user explicitly requests another language, **RESPOND IN KOREAN**.

[CITATION RULES]
- Assign each unique citation a single number, and number sources sequentially (1, 2, 3, 4, ...) as <CITATION_NUMBER> in you text
- Append the citation for the supporting source(s) after EVERY CORRESPONDING TEXT BLOCK(single sentence or a paragraph, ...), in-line (not collected at the end), using exactly this format: <CITATION_NUMBER>; if multiple sources support a sentence, list all citations in source order separated by spaces.
- <DOC_ID> and <HEADING_ID> is shown along with each context.
- Insert citation information in markdown format style as shown below.

[CITATION FORMAT]
- [<CITATION_NUMBER>](btn:<DOC_ID>/<HEADING_ID>)
- ex) [2](btn:d4b0de35-6b77-4971-a07e-af3dbe31e41d/8855ed9e-9128-4774-a538-e5a53b843393)

[CORRECTLY FORMATTED ANSWER EXAMPLE]
국외여행자가 3급 이하인 경우에는 2급 직원에 상당하는 여비가 지급됩니다. 즉, 실직급이 3급 이하여도 여비 지급은 2급 기준을 적용합니다[1](btn:38ffed71-d635-4a4e-9b17-3de8afa4b80a/78206468-e213-4630-a283-2b49b7faa7df).\n\n숙박비의 지급과 관련해서는 기본적으로 항공여행의 경우 숙박비는 지급하지 않으며, 필요한 경우에만 식비를 따로 지급합니다. 그러나 천재지변이나 그 밖의 부득이한 사유로 인해 육지에서 숙박이 필요할 때에는 숙박비를 지급할 수 있습니다[2](btn:38ffed71-d635-4a4e-9b17-3de8afa4b80a/b2acee2d-674a-44d0-9182-e8fc38c45bd5).
"""

GENERATE_RAG_ANSWER_USER = """[Context]
{documents}

[User's question]
{input}"""

SINGLE_SUMMARY_SYSTEM = """Summarize user's input, in a single short sentence.
Unless a user explicitly requests another language, **RESPOND IN KOREAN**."""

MERGE_SUMMARY_SYSTEM = """Refer to list of summaries and re-summarize what these documents are about, in a single paragraph.
Unless a user explicitly requests another language, **RESPOND IN KOREAN**.
"""