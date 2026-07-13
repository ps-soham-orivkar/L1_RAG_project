# tools.py
# Standard Python tools: Attendance calculator and eligibility checker.

def calculate_attendance(total_classes, attended_classes):
    """
    Calculates the attendance percentage given total classes and attended classes.
    """
    try:
        total = int(total_classes)
        attended = int(attended_classes)
        if total <= 0:
            return "Error: Total classes must be greater than 0."
        if attended > total:
            return "Error: Attended classes cannot be greater than total classes."
            
        percentage = (attended / total) * 100
        return f"Your attendance is {percentage:.2f}%."
    except ValueError:
        return "Error: Please provide valid integers for classes."

def check_eligibility(attendance_percentage):
    """
    Checks if a student is eligible based on their attendance percentage.
    The required minimum attendance is 75%.
    """
    try:
        attendance = float(attendance_percentage)
        required_percentage = 75.0
        if attendance >= required_percentage:
            return f"Eligible: Your attendance ({attendance:.2f}%) meets the required {required_percentage}%."
        else:
            return f"Not Eligible: Your attendance ({attendance:.2f}%) is below the required {required_percentage}%."
    except ValueError:
        return "Error: Please provide a valid float for attendance percentage."

def route_query_to_tool(query, retriever=None):
    """
    Simple routing logic to determine if a query should be handled by a tool instead of RAG.
    Returns the tool response if applicable, else None.
    """
    query_lower = query.lower()
    
    if "calculate attendance" in query_lower:
        parts = query_lower.replace("calculate attendance", "").strip().split()
        if len(parts) >= 2:
            return calculate_attendance(parts[0], parts[1])
        else:
            return "To calculate attendance, please provide total classes and attended classes. (e.g., 'calculate attendance 40 30')"
            
    if "check eligibility" in query_lower:
        parts = query_lower.replace("check eligibility", "").strip().split()
        if len(parts) >= 1:
            return check_eligibility(parts[0])
        else:
            return "To check eligibility, please provide your attendance percentage. (e.g., 'check eligibility 80')"
            
    # Custom tool logic for PDF querying issues
    if retriever and ("list the topics" in query_lower or "list topics" in query_lower or "main topics" in query_lower):
        def _stream_topics():
            try:
                from chatbot import get_llm
                llm = get_llm()
                all_content = "\n".join([doc.page_content for doc in retriever.documents[:15]])
                prompt = f"Please list the main topics from the following document content:\n\n{all_content}"
                response_stream = llm.stream(prompt)
                accumulated = ""
                for chunk in response_stream:
                    accumulated += chunk.content
                    yield accumulated
            except Exception as e:
                yield f"Error extracting topics: {str(e)}"
        return _stream_topics()
            
    import re
    page_match = re.search(r'explain\s*(the\s*)?page\s*(\d+)', query_lower)
    if retriever and page_match:
        def _stream_page():
            try:
                page_num = int(page_match.group(2))
                page_chunks = [doc.page_content for doc in retriever.documents if doc.metadata.get('page') == page_num - 1]
                if not page_chunks:
                    page_chunks = [doc.page_content for doc in retriever.documents if doc.metadata.get('page') == page_num]
                    
                if not page_chunks:
                    yield f"Sorry, I couldn't find any content for page {page_num} in the uploaded documents."
                    return
                    
                from chatbot import get_llm
                llm = get_llm()
                page_content = "\n".join(page_chunks)
                prompt = f"Please explain the following content from page {page_num} in simple words:\n\n{page_content}"
                response_stream = llm.stream(prompt)
                accumulated = ""
                for chunk in response_stream:
                    accumulated += chunk.content
                    yield accumulated
            except Exception as e:
                yield f"Error explaining page: {str(e)}"
        return _stream_page()

    return None
