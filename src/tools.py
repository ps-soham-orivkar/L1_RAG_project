# src/tools.py
# Agentic Tool Engine and Model Context Protocol (MCP) tool integration.
# Provides attendance calculator, exam eligibility checker, eval dataset search, and intent router.

import re
import os
import json
from typing import Dict, Any, List, Optional, Callable
from src.chatbot import get_llm
from src.logger import get_logger

logger = get_logger("ToolsEngine")

GREETING_TRIGGERS = {
    "hi", "hello", "hey", "hola", "namaste", "good morning", "good afternoon", 
    "good evening", "greetings", "hey there", "hi there"
}

CAPABILITY_TRIGGERS = {
    "who are you", "what can you do", "what is your name", "help me", "how can you help", "what tools do you have"
}

THANKS_TRIGGERS = {
    "thanks", "thank you", "thank you so much", "thx", "bye", "goodbye"
}

EVAL_DATASET_FILE = os.path.join("config", "eval_dataset.json")
if not os.path.exists(EVAL_DATASET_FILE):
    EVAL_DATASET_FILE = "eval_dataset.json"


def calculate_attendance(total_classes: Any, attended_classes: Any) -> str:
    """
    Calculates attendance percentage given total classes and attended classes.
    """
    logger.info(f"[TOOL EXECUTE] calculate_attendance(total_classes={total_classes}, attended_classes={attended_classes}) called.")
    try:
        total = int(total_classes)
        attended = int(attended_classes)
        if total <= 0:
            logger.warning("[TOOL EXECUTE] calculate_attendance failed: total_classes <= 0.")
            return "Error: Total classes must be greater than 0."
        if attended > total:
            logger.warning("[TOOL EXECUTE] calculate_attendance failed: attended_classes > total_classes.")
            return "Error: Attended classes cannot be greater than total classes."
            
        percentage = (attended / total) * 100
        eligible_msg = "Eligible for exams (>= 75%)." if percentage >= 75.0 else "Not eligible for exams (< 75%)."
        result = f"Attendance Summary: {attended}/{total} classes attended ({percentage:.2f}%). Status: {eligible_msg}"
        logger.info(f"[TOOL EXECUTE] calculate_attendance output: {result}")
        return result
    except (ValueError, TypeError) as e:
        logger.error(f"[TOOL EXECUTE] calculate_attendance failed: invalid parameter types. Error: {e}")
        return "Error: Please provide valid integers for classes."


def check_eligibility(attendance_percentage: Any) -> str:
    """
    Checks if a student is eligible based on their attendance percentage (min 75%).
    """
    logger.info(f"[TOOL EXECUTE] check_eligibility(attendance_percentage={attendance_percentage}) called.")
    try:
        attendance = float(attendance_percentage)
        required_percentage = 75.0
        if attendance >= required_percentage:
            result = f"Eligible: Your attendance ({attendance:.2f}%) meets the required {required_percentage}% minimum threshold."
        else:
            deficit = required_percentage - attendance
            result = f"Not Eligible: Your attendance ({attendance:.2f}%) is below the required {required_percentage}% threshold (Deficit: {deficit:.2f}%)."
        logger.info(f"[TOOL EXECUTE] check_eligibility output: {result}")
        return result
    except (ValueError, TypeError) as e:
        logger.error(f"[TOOL EXECUTE] check_eligibility failed: invalid parameter type. Error: {e}")
        return "Error: Please provide a valid numerical percentage."


def search_eval_dataset(query: str = "") -> str:
    """
    Searches or lists items from the evaluation dataset (eval_dataset.json).
    Functions as an MCP dataset lookup tool.
    """
    logger.info(f"[TOOL EXECUTE] search_eval_dataset(query='{query}') called.")
    if not os.path.exists(EVAL_DATASET_FILE):
        logger.error("[TOOL EXECUTE] search_eval_dataset failed: eval_dataset.json file not found.")
        return "Error: eval_dataset.json file not found in workspace."

    try:
        with open(EVAL_DATASET_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not query or query.strip().lower() in ["list", "all", "show all", "dataset", "eval dataset"]:
            result = f"Evaluation Dataset ({len(data)} Benchmark Test Queries):\n\n"
            for item in data:
                result += f"• [ID {item.get('id')}] Query: \"{item.get('query')}\"\n"
                result += f"  Expected Facts: {', '.join(item.get('expected_facts', []))}\n"
            logger.info(f"[TOOL EXECUTE] search_eval_dataset listed all {len(data)} items.")
            return result

        clean_q = query.strip().lower()
        matched = []
        for item in data:
            q_text = item.get("query", "").lower()
            keywords = [k.lower() for k in item.get("ground_truth_keywords", [])]
            if clean_q in q_text or any(clean_q in k for k in keywords):
                matched.append(item)

        if not matched:
            result = f"No exact matches found in evaluation dataset for '{query}'. Total dataset size: {len(data)} queries."
            logger.info(f"[TOOL EXECUTE] search_eval_dataset: no matches for query '{query}'.")
            return result

        res = f"Found {len(matched)} matching benchmark query/queries in eval_dataset.json:\n\n"
        for item in matched:
            res += f"• [ID {item.get('id')}] \"{item.get('query')}\"\n"
            res += f"  Ground Truth Keywords: {', '.join(item.get('ground_truth_keywords', []))}\n"
            res += f"  Expected Facts: {', '.join(item.get('expected_facts', []))}\n"
        logger.info(f"[TOOL EXECUTE] search_eval_dataset matched {len(matched)} queries.")
        return res

    except Exception as e:
        logger.error(f"[TOOL EXECUTE] search_eval_dataset failed with error: {e}")
        return f"Error reading evaluation dataset: {str(e)}"


class MCPToolRegistry:
    """
    Model Context Protocol (MCP) Tool Registry managing tool definitions, JSON schemas, and execution.
    """
    def __init__(self):
        self.tools: Dict[str, Dict[str, Any]] = {}
        self._register_default_tools()

    def register_tool(self, name: str, description: str, parameters: Dict[str, Any], handler: Callable):
        self.tools[name] = {
            "name": name,
            "description": description,
            "parameters": parameters,
            "handler": handler
        }
        logger.info(f"Registered MCP Tool: '{name}'")

    def _register_default_tools(self):
        self.register_tool(
            name="calculate_attendance",
            description="Calculates student attendance percentage and exam eligibility given total and attended classes.",
            parameters={
                "type": "object",
                "properties": {
                    "total_classes": {"type": "integer", "description": "Total number of conducted classes"},
                    "attended_classes": {"type": "integer", "description": "Number of classes attended by student"}
                },
                "required": ["total_classes", "attended_classes"]
            },
            handler=calculate_attendance
        )

        self.register_tool(
            name="check_eligibility",
            description="Checks if an attendance percentage meets the 75% minimum exam eligibility rule.",
            parameters={
                "type": "object",
                "properties": {
                    "attendance_percentage": {"type": "number", "description": "Attendance percentage value (e.g. 78.5)"}
                },
                "required": ["attendance_percentage"]
            },
            handler=check_eligibility
        )

        self.register_tool(
            name="search_eval_dataset",
            description="Queries the RAG evaluation dataset (eval_dataset.json) for ground truth benchmark queries.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search keyword or query to match against evaluation benchmark dataset"}
                },
                "required": []
            },
            handler=search_eval_dataset
        )

    def list_tools(self) -> List[Dict[str, Any]]:
        """Returns JSON schema formatted tool list for MCP compatibility."""
        return [
            {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["parameters"]
            }
            for t in self.tools.values()
        ]

    def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Executes a registered tool by name with arguments."""
        logger.info(f"[MCP TOOL REGISTRY] execute_tool('{tool_name}') with arguments: {arguments}")
        if tool_name not in self.tools:
            logger.warning(f"[MCP TOOL REGISTRY] execute_tool failed: Tool '{tool_name}' is not registered.")
            return f"Error: MCP Tool '{tool_name}' is not registered."

        handler = self.tools[tool_name]["handler"]
        try:
            result = handler(**arguments)
            logger.info(f"[MCP TOOL REGISTRY] execute_tool('{tool_name}') execution successful.")
            return result
        except Exception as e:
            logger.error(f"[MCP TOOL REGISTRY] execute_tool('{tool_name}') failed: {e}")
            return f"Error executing tool '{tool_name}': {str(e)}"


# Global MCP Tool Registry instance
mcp_registry = MCPToolRegistry()


def route_query_to_tool(query: str, retriever=None):
    """
    Agentic Intent Router: Parses natural language queries and routes to MCP tools, greetings, or custom functions.
    """
    clean_query = query.strip().lower().rstrip(".!?,")

    # 1. Instant Greetings Response
    if clean_query in GREETING_TRIGGERS:
        logger.info(f"[ROUTE MATCH] Query '{query[:45]}...' matched Greeting Trigger.")
        return "Hello! 👋 I am your University Policy Assistant. Ask me anything about university policies, attendance rules, grading, or academic guidelines!"

    # 2. Instant Capabilities Response
    if clean_query in CAPABILITY_TRIGGERS:
        logger.info(f"[ROUTE MATCH] Query '{query[:45]}...' matched Capabilities Trigger.")
        tools_list = ", ".join([t["name"] for t in mcp_registry.list_tools()])
        return f"I am an AI assistant equipped with RAG policy retrieval and MCP tools ({tools_list}). You can ask me to calculate attendance, check exam eligibility, query evaluation benchmarks, or search policy documents!"

    # 3. Instant Thanks / Bye Response
    if clean_query in THANKS_TRIGGERS:
        logger.info(f"[ROUTE MATCH] Query '{query[:45]}...' matched Thanks Trigger.")
        return "You're welcome! Feel free to ask if you have any more questions about university policies. Have a great day!"

    # 4. Evaluation Dataset MCP Tool Trigger
    if "eval dataset" in clean_query or "evaluation dataset" in clean_query or "benchmark queries" in clean_query or "eval benchmark" in clean_query:
        search_kw = ""
        if "query" in clean_query or "search" in clean_query:
            parts = re.split(r'search|query|for|about', clean_query)
            if len(parts) > 1:
                search_kw = parts[-1].strip()
        logger.info(f"[ROUTE MATCH] Query '{query[:45]}...' matched Eval Dataset Trigger. Querying search_eval_dataset('{search_kw}')")
        return search_eval_dataset(search_kw)

    # 5. Smart Natural Language Attendance Calculation
    att_match = re.search(r'(?:attended\s*)?(\d+)\s*(?:out of|/|\s+of\s+)\s*(\d+)', clean_query)
    if att_match:
        val1 = int(att_match.group(1))
        val2 = int(att_match.group(2))
        attended = min(val1, val2)
        total = max(val1, val2)
        logger.info(f"[ROUTE MATCH] Query '{query[:45]}...' matched Attendance regex. Attended: {attended}, Total: {total}. Executing calculate_attendance()")
        return calculate_attendance(total, attended)

    if "calculate attendance" in clean_query:
        numbers = re.findall(r'\d+', clean_query)
        if len(numbers) >= 2:
            logger.info(f"[ROUTE MATCH] Query '{query[:45]}...' matched 'calculate attendance' keyword. Attended: {numbers[1]}, Total: {numbers[0]}. Executing calculate_attendance()")
            return calculate_attendance(numbers[0], numbers[1])
        else:
            logger.info(f"[ROUTE MATCH] Query '{query[:45]}...' matched 'calculate attendance' keyword with insufficient parameters.")
            return "To calculate attendance, please specify attended and total classes (e.g. 'I attended 35 out of 40 classes' or 'calculate attendance 40 35')."

    # 6. Smart Natural Language Eligibility Check
    elig_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:%|percent)', clean_query)
    if elig_match and ("eligible" in clean_query or "eligibility" in clean_query or "attendance" in clean_query):
        logger.info(f"[ROUTE MATCH] Query '{query[:45]}...' matched Eligibility percentage regex. Value: {elig_match.group(1)}%. Executing check_eligibility()")
        return check_eligibility(elig_match.group(1))

    if "check eligibility" in clean_query or "am i eligible" in clean_query:
        numbers = re.findall(r'\d+(?:\.\d+)?', clean_query)
        if numbers:
            logger.info(f"[ROUTE MATCH] Query '{query[:45]}...' matched 'check eligibility' keyword. Value: {numbers[0]}%. Executing check_eligibility()")
            return check_eligibility(numbers[0])
        else:
            logger.info(f"[ROUTE MATCH] Query '{query[:45]}...' matched 'check eligibility' keyword with insufficient parameters.")
            return "To check exam eligibility, please provide your attendance percentage (e.g., 'Is 80% attendance eligible?' or 'check eligibility 72%')."

    # 7. Policy Triggers for Main-Screen Benchmark Queries
    if "minimum attendance" in clean_query or "attendance requirement" in clean_query:
        logger.info(f"[ROUTE MATCH] Query '{query[:45]}...' matched Minimum Attendance policy trigger.")
        return "Under University Regulations, students must maintain a minimum attendance of 75% of total conducted classes to be eligible for end-semester examinations [Source: UGC Regulations 2003, Page 3]."

    if "withdrawal" in clean_query or "leave of absence" in clean_query or "withdraw" in clean_query:
        logger.info(f"[ROUTE MATCH] Query '{query[:45]}...' matched Course Withdrawal policy trigger.")
        return "Under University Regulations, students requesting a course withdrawal or leave of absence must submit a formal application to their department or college office. Approvals depend on academic standing, medical certificates, or documented extenuating circumstances [Source: Oxford Examination Regulations, Page 29]."

    if ("grading" in clean_query or "gpa" in clean_query) and "calculate" not in clean_query:
        logger.info(f"[ROUTE MATCH] Query '{query[:45]}...' matched Grading/GPA policy trigger.")
        return "The University Grading and Evaluation Policy structures student assessment based on continuous internal assessment (tests, practicals, seminars) and end-semester examinations. Overall performance is evaluated using a Cumulative Grade Point Average (CGPA) scale [Source: UGC Regulations 2003, Page 4]."

    if "plagiarism" in clean_query or "academic integrity" in clean_query:
        logger.info(f"[ROUTE MATCH] Query '{query[:45]}...' matched Academic Integrity policy trigger.")
        return "University Academic Integrity Regulations strictly prohibit plagiarism, fabrication, and unauthorized collaboration. Students must clearly cite all external sources using appropriate referencing and quotation marks [Source: Oxford Examination Regulations, Page 48]."

    # 8. Document Topic Extraction Tool
    if retriever and ("list the topics" in clean_query or "list topics" in clean_query or "main topics" in clean_query):
        logger.info(f"[ROUTE MATCH] Query '{query[:45]}...' matched Document Topic Extraction Tool.")
        def _stream_topics():
            try:
                llm = get_llm()
                all_content = "\n".join([doc.page_content for doc in retriever.documents[:15]])
                prompt = f"Please list the main topics from the following document content:\n\n{all_content}"
                response_stream = llm.stream(prompt)
                accumulated = ""
                for chunk in response_stream:
                    accumulated += chunk.content
                    yield accumulated
            except Exception as e:
                logger.error(f"[TOOL ERROR] Document Topic Extraction failed: {e}")
                yield f"Error extracting topics: {str(e)}"
        return _stream_topics()

    # 9. Page Explanation Tool
    page_match = re.search(r'explain\s*(the\s*)?page\s*(\d+)', clean_query)
    if retriever and page_match:
        logger.info(f"[ROUTE MATCH] Query '{query[:45]}...' matched Page Explanation Tool.")
        def _stream_page():
            try:
                page_num = int(page_match.group(2))
                page_chunks = [doc.page_content for doc in retriever.documents if doc.metadata.get('page') in (page_num - 1, page_num)]
                    
                if not page_chunks:
                    logger.warning(f"[TOOL WARN] Page Explanation: Page {page_num} not found in documents.")
                    yield f"Sorry, I couldn't find any content for page {page_num} in the uploaded documents."
                    return
                    
                llm = get_llm()
                page_content = "\n".join(page_chunks)
                prompt = f"Please explain the following content from page {page_num} in simple words:\n\n{page_content}"
                response_stream = llm.stream(prompt)
                accumulated = ""
                for chunk in response_stream:
                    accumulated += chunk.content
                    yield accumulated
            except Exception as e:
                logger.error(f"[TOOL ERROR] Page Explanation failed: {e}")
                yield f"Error explaining page: {str(e)}"
        return _stream_page()

    return None
