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

def route_query_to_tool(query):
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
            
    return None
