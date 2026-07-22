# tests/test_tools_mcp.py
# Automated verification test script for Agentic Tools Engine and MCP Tool Registry.

import json
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.tools import (
    calculate_attendance,
    check_eligibility,
    search_eval_dataset,
    mcp_registry,
    route_query_to_tool
)

def test_attendance_tool():
    print("=== Testing Attendance Calculator Tool ===")
    res1 = calculate_attendance(40, 30)
    assert "75.00%" in res1, f"Expected 75.00%, got '{res1}'"
    print(f"[PASS] Exact Attendance Calc: {res1}")

    # Test natural language routing
    nl_res = route_query_to_tool("I attended 36 out of 45 classes, what is my percentage?")
    assert "80.00%" in nl_res, f"Expected 80.00%, got '{nl_res}'"
    print(f"[PASS] Natural Language Routing (36/45): {nl_res}")

def test_eligibility_tool():
    print("\n=== Testing Exam Eligibility Tool ===")
    res1 = check_eligibility(80)
    assert "Eligible" in res1 and "80.00%" in res1
    print(f"[PASS] Eligible Check: {res1}")

    res2 = check_eligibility(70)
    assert "Not Eligible" in res2 and "5.00%" in res2
    print(f"[PASS] Deficit Check: {res2}")

    # Test natural language routing
    nl_res = route_query_to_tool("My attendance is 78%, am I eligible for exams?")
    assert "Eligible" in nl_res and "78.00%" in nl_res
    print(f"[PASS] Natural Language Routing (78%): {nl_res}")

def test_eval_dataset_tool():
    print("\n=== Testing Eval Dataset MCP Tool ===")
    all_data = search_eval_dataset()
    assert "Evaluation Dataset" in all_data and "Benchmark Test Queries" in all_data
    print(f"[PASS] Dataset Listing verified.")

    query_match = search_eval_dataset("academic integrity")
    assert "plagiarism" in query_match.lower()
    print(f"[PASS] Dataset Search Keyword Match verified:\n{query_match}")

    # Test natural language routing for eval dataset
    nl_res = route_query_to_tool("Show eval dataset benchmark queries")
    assert "Evaluation Dataset" in nl_res
    print("[PASS] Natural Language Routing for Eval Dataset verified!")

def test_mcp_registry():
    print("\n=== Testing MCP Registry & Execution Engine ===")
    tools_list = mcp_registry.list_tools()
    tool_names = [t["name"] for t in tools_list]
    assert "calculate_attendance" in tool_names
    assert "check_eligibility" in tool_names
    assert "search_eval_dataset" in tool_names
    print(f"[PASS] MCP Tool Schemas listed: {tool_names}")

    exec_res = mcp_registry.execute_tool("calculate_attendance", {"total_classes": 50, "attended_classes": 40})
    assert "80.00%" in exec_res
    print(f"[PASS] Direct MCP Execution result: {exec_res}")

if __name__ == "__main__":
    test_attendance_tool()
    test_eligibility_tool()
    test_eval_dataset_tool()
    test_mcp_registry()
    print("\n[SUCCESS] All Agentic Tools & MCP Registry unit tests passed successfully!")
