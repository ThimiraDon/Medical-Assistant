import re
from markupsafe import Markup

def format_documents(docs):

    formatted_docs = []

    for i, doc in enumerate(docs):
        formatted_docs.append(
            f"Document {i+1}:\n{doc.page_content}"
        )

    return "\n\n".join(formatted_docs)

def format_response_html(raw_text):
    """
    Converts numbered plain text into HTML with proper spacing and bullets.
    """
    # Split text into numbered sections (1., 2., 3., etc.)
    sections = re.split(r'(\d+\.\s)', raw_text)
    formatted = ""
    skip_next = False

    for i in range(len(sections)):
        if skip_next:
            skip_next = False
            continue

        if re.match(r'\d+\.\s', sections[i]):
            if i+1 < len(sections):
                section_text = sections[i+1].strip()
                if ':' in section_text:
                    heading, body = section_text.split(':', 1)
                    formatted += f"<b>{sections[i]}{heading}:</b> {body.strip()}<br><br>"
                else:
                    formatted += f"<b>{sections[i]}{section_text}</b><br><br>"
                skip_next = True
        else:
            formatted += sections[i].strip() + "<br><br>"

    # Convert '-' bullets to proper HTML list items
    if '-' in formatted:
        formatted = re.sub(r'(?m)^-\s+', r'<li>', formatted)
        formatted = '<ul>' + formatted + '</ul>'

    return Markup(formatted)