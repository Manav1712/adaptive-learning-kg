"""
Pre-configured student profiles for demo purposes.

To switch between profiles, change the ACTIVE_PROFILE variable below.

Test Scenarios:
1. Trig Equations: "I want to learn about solving trigonometric equations in quadratic form"
2. Complex Numbers: "I want to learn about plotting complex numbers"
3. Right Triangles: "I want to learn about solving right triangle problems"
4. Sum/Difference Identities: "I want to learn about sum and difference identities"
5. Calculus basics: "I want to learn derivatives"
"""

# ============================================
# CHANGE THIS VALUE TO SWITCH PROFILES
# 1 = Strong Student (high proficiency)
# 2 = Weak Student (low proficiency)
# ============================================
ACTIVE_PROFILE = 2


# Profile 1: Strong Student
# Expected behavior: Skips prerequisites, focuses on main topic
# session_guidance: "Focus primarily on [topic] without extensive prerequisite review"
STRONG_STUDENT = {
    "lo_mastery": {
        # --- Calculus Topics ---
        "Derivatives": 0.9,
        "Limits": 0.85,
        "Integrals": 0.75,
        "Chain Rule": 0.8,
        "Product Rule": 0.8,
        "Implicit Differentiation": 0.75,
        "Functions": 0.9,
        
        # --- Trig Equations (Scenario 1, 2, 3) ---
        "Solving Linear Trigonometric Equations in Sine and Cosine": 0.85,
        "Solving Trigonometric Equations": 0.9,
        "Solving Equations Involving a Single Trigonometric Function": 0.8,
        "Solving Trigonometric Equations Using Fundamental Identities": 0.75,
        "Solving Trigonometric Equations with Multiple Angles": 0.8,
        "Solve Trigonometric Equations Using a Calculator": 0.7,
        
        # --- Complex Numbers (Scenario 2) ---
        "Writing Complex Numbers in Polar Form": 0.85,
        "Finding Products of Complex Numbers in Polar Form": 0.8,
        "Finding Quotients of Complex Numbers in Polar Form": 0.75,
        "Polar Form of Complex Numbers": 0.8,
        
        # --- Sum/Difference Identities (Scenario 4) ---
        "Using the Sum and Difference Formulas to Verify Identities": 0.85,
        "Using the Sum and Difference Formulas for Cosine": 0.9,
        "Using the Sum and Difference Formulas for Sine": 0.85,
    }
}


# Profile 2: Weak Student
# Expected behavior: Includes prerequisite reviews before main topic
# session_guidance: "Begin the session with focused refreshers on [prereqs]..."
WEAK_STUDENT = {
    "lo_mastery": {
        # --- Calculus Topics ---
        "Derivatives": 0.3,
        "Limits": 0.4,
        "Integrals": 0.2,
        "Chain Rule": 0.1,
        "Product Rule": 0.25,
        "Implicit Differentiation": 0.15,
        "Functions": 0.35,
        
        # --- Trig Equations (Scenario 1, 2, 3) ---
        "Solving Linear Trigonometric Equations in Sine and Cosine": 0.3,
        "Solving Trigonometric Equations": 0.25,
        "Solving Equations Involving a Single Trigonometric Function": 0.2,
        "Solving Trigonometric Equations Using Fundamental Identities": 0.15,
        "Solving Trigonometric Equations with Multiple Angles": 0.2,
        "Solve Trigonometric Equations Using a Calculator": 0.1,
        
        # --- Complex Numbers (Scenario 2) ---
        "Writing Complex Numbers in Polar Form": 0.3,
        "Finding Products of Complex Numbers in Polar Form": 0.2,
        "Finding Quotients of Complex Numbers in Polar Form": 0.15,
        "Polar Form of Complex Numbers": 0.25,
        
        # --- Sum/Difference Identities (Scenario 4) ---
        "Using the Sum and Difference Formulas to Verify Identities": 0.3,
        "Using the Sum and Difference Formulas for Cosine": 0.25,
        "Using the Sum and Difference Formulas for Sine": 0.2,
    }
}


def get_active_profile() -> dict:
    """
    Returns the currently active student profile based on ACTIVE_PROFILE setting.
    
    Returns:
        Dict with "lo_mastery" scores for the selected profile.
    """
    if ACTIVE_PROFILE == 1:
        return STRONG_STUDENT
    return WEAK_STUDENT


def get_profile_name() -> str:
    """Returns the name of the currently active profile."""
    return "Strong Student" if ACTIVE_PROFILE == 1 else "Weak Student"
