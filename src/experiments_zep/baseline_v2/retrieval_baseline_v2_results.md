BASELINE V2 KNOWLEDGE GRAPH — EVALUATION REPORT (Clean)

Graph: baseline_v2_experiment

1) Graph Statistics
- Nodes: 743
- Edges: 968
- Node–Edge Ratio: 1.30

2) Retrieval Performance (Limited)
- Test queries returned exactly 5 nodes/5 edges each.
- Note: Retrieval results are capped (limit=5) and may be additionally rate-limited; these numbers are not a measure of coverage or quality.

3) Relationship Constraint Analysis
- Total relationships: 968
- Target relationships (PREREQUISITE_OF, PART_OF, ASSESSED_BY): 238
- Noise relationships (everything else): 730
- Constraint effectiveness: 24.6%
- Top relationship types (by count):
  - PART_OF (148)
  - EQUALS (106)
  - ASSESSED_BY (90)
  - EQUIVALENT_TO (55)
  - "The Limit of a Function" (53)
  - HAS_PROPERTY (36)
  - IS_A (35)

4) Content Coverage (Search-Limited Proxy)
- LO-related nodes found via search: 50
- Content type distribution from summaries: concept (61), example (99), problem (116), try_it (13), other (454)
- Unit/Chapter coverage not reliably present in metadata in this run.

5) Contextual Retrieval (Search-Limited)
- LO IDs tested: 1867, 1868, 1869, 1870, 1872 → each returned 10 nodes/10 edges (due to query limits). Treat as functionality check, not quality.

6) Why So Many Edge Types?
- Open-vocabulary extraction: Without a strict ontology, the extractor names many relation variants (e.g., EQUALS, IS_EQUAL_TO) and domain phrases (e.g., "The Limit of a Function").
- Brief schema hints are advisory only: They nudge, but do not constrain extraction.
- Content diversity: Math statements, steps, and properties surface varied predicates.
- No enforced ontology: Since `set_ontology()` was not applied, extraction remained unconstrained, inflating the unique edge-type count (169).

   📊 NODE ANALYSIS (743 total)
     Average Summary Length: 411 characters
     Labels Distribution:
       • Entity         : 743
     Content Type Distribution:
       • other     : 454 nodes
       • problem   : 116 nodes
       • example   :  99 nodes
       • concept   :  61 nodes
       • try_it    :  13 nodes

   🔗 EDGE ANALYSIS (968 total)
     Average Fact Length: 60 characters
     Relationship Types (169 total):
       • PART_OF                  : 148
       • EQUALS                   : 106
       • ASSESSED_BY              :  90
       • EQUIVALENT_TO            :  55
       • The Limit of a Function  :  53
       • HAS_PROPERTY             :  36
       • IS_A                     :  35
       • HAS_DOMAIN               :  18
       • DERIVED_FROM             :  18
       • RELATED_TO               :  15
       • EVALUATES_TO             :  14
       • HAS_RANGE                :  13
       • CALCULATED_AS            :  13
       • HAS_VALUE                :  13
       • APPROACHES               :  11
       • SATISFIES                :   9
       • IS_GRAPH_OF              :   9
       • CALCULATED_USING         :   7
       • DESCRIBES                :   7
       • PASSES_THROUGH           :   7
       • SOLVES_FOR               :   7
       • IS_LESS_THAN             :   7
       • USES                     :   6
       • CALCULATED_FOR           :   6
       • DETERMINES               :   6
       • INTERSECTS               :   5
       • MEASURED_IN              :   5
       • IS_GREATER_THAN_OR_EQUAL_TO:   5
       • USED_TO_SIMPLIFY         :   5
       • APPLIED_IN_STEP          :   4
       • TRANSFORMED_INTO         :   4
       • IS                       :   4
       • IMPLIES                  :   4
       • TRANSFORMATION_OF        :   4
       • HAS_ZERO                 :   4
       • DERIVED_INTO             :   4
       • IS_TRANSFORMATION_OF     :   4
       • DEFINED_BY               :   4
       • CAUSES                   :   4
       • CORRESPONDS_TO           :   4
       • TRANSFORMS               :   4
       • CONVERSION_FACTOR_FOR    :   4
       • CALCULATES               :   3
       • IS_ZERO_OF               :   3
       • HAS_DENOMINATOR          :   3
       • COMPARED_TO              :   3
       • HAS_COST                 :   3
       • Limits                   :   3
       • IS_DOMAIN_OF             :   3
       • IS_SYMMETRIC_ABOUT       :   3
       • APPLIED_TO               :   3
       • IS_CALCULATED_AS         :   3
       • HAS_CONDITION            :   3
       • NOT_EQUALS               :   3
       • DETERMINE_DOMAIN_OF      :   2
       • REQUIRES_CONDITION       :   2
       • IS_PART_OF               :   2
       • IS_NEVER_ZERO            :   2
       • REQUIRES                 :   2
       • VERIFIES                 :   2
       • EVALUATED_AT             :   2
       • INDICATES                :   2
       • LOCATED_AT               :   2
       • EXISTS                   :   2
       • BOUNDED_BY               :   2
       • COMPARED_WITH            :   2
       • USED_TO_EVALUATE         :   2
       • EXISTS_FOR               :   2
       • IS_INVERSE_OF            :   2
       • IS_PART_OF_CALCULATION   :   2
       • IS_ON_GRAPH_OF           :   2
       • APPROXIMATELY_EQUALS     :   2
       • DEFINED_AS               :   2
       • CONVERTS_TO              :   2
       • APPLIES                  :   2
       • Exponential and Logarithmic Functions:   2
       • IS_RANGE_OF              :   2
       • ESTIMATES                :   2
       • FAILS_TO_HOLD            :   2
       • PREDICTED_REVENUE_FOR_PRICE:   2
       • HAS_POINT                :   2
       • MEASURES                 :   2
       • CONTINUOUS_OVER          :   2
       • IS_CALCULATED_BY         :   2
       • APPLIES_TO               :   2
       • IS_ABOUT                 :   2
       • INTERPRETED_AS           :   2
       • PROPERTY_OF              :   2
       • FACTORED_INTO            :   2
       • SOLVES                   :   2
       • APPROXIMATES             :   1
       • HAS_CHAPTER              :   1
       • HAS_COST_FOR_FIRST_UNIT  :   1
       • FOR_ALL                  :   1
       • rate of change to f(x) = x^2 at x = 1:   1
       • CANNOT_BE_WRITTEN_USING_FORMULA_INVOLVING:   1
       • ESTIMATED_BY             :   1
       • TESTED_BY                :   1
       • MAXIMIZES                :   1
       • EXISTS_SUCH_THAT         :   1
       • HAS_SLOPE                :   1
       • HAS_UNIT                 :   1
       • IS_REWRITTEN_AS          :   1
       • IS_SAME_AS               :   1
       • CAN_BE_REWRITTEN_IN_TERMS_OF:   1
       • MUST_BE_NON_NEGATIVE     :   1
       • HAS_NO_SOLUTION_IF       :   1
       • HAS_HEIGHT               :   1
       • IS_EQUAL_TO              :   1
       • IS_AN_EXAMPLE_OF         :   1
       • MORE_INTENSE_THAN        :   1
       • CONFIRMS                 :   1
       • IS_DEFINED_FOR           :   1
       • IS_FIGURE_FOR            :   1
       • IS_ONE_TO_ONE_ON_RESTRICTED_DOMAIN:   1
       • EVALUATED_TO             :   1
       • IS_CLOSE_TO              :   1
       • HAS_DISCONTINUITY        :   1
       • CAN_BE_REWRITTEN_USING   :   1
       • DOES_NOT_EQUAL           :   1
       • HAS_LIMIT                :   1
       • DEVELOPED                :   1
       • IS_REQUIRED_FOR          :   1
       • HAS_EQUATION             :   1
       • IS_TRUE_WHEN             :   1
       • MORE_SEVERE_THAN         :   1
       • HAS_ANGLE                :   1
       • USES_FORMULA             :   1
       • SATISFIES_EQUATION       :   1
       • REQUIRES_CHECK_FOR       :   1
       • HAS_BASE                 :   1
       • INTERCHANGING            :   1
       • IS_ZERO_FOR              :   1
       • IS_NOT_EQUAL_TO          :   1
       • SUMMED_TO                :   1
       • IS_SHOWN_IN              :   1
       • SUBSTITUTES              :   1
       • IS_THE_DOMAIN            :   1
       • ESTIMATED_AS             :   1
       • IS_CALCULATED_BY_SUBSTITUTING:   1
       • FORMULA_FOR              :   1
       • DEFINES                  :   1
       • RESTRICTED_BY            :   1
       • Functions and Graphs     :   1
       • CLOSER_APPROXIMATION     :   1
       • MEASURED_ON              :   1
       • IS_GREATER_THAN          :   1
       • HAS_RATE_OF_CHANGE_AT    :   1
       • HAS_VERTICAL_ASYMPTOTE   :   1
       • RESTRICTED_TO            :   1
       • HAS_FORM                 :   1
       • IS_FORMULA_FOR           :   1
       • CONTINUOUS_FOR           :   1
       • CAN_BE_EVALUATED_USING   :   1
       • DENOTES                  :   1
       • IS_CONTINUOUS_OVER       :   1
       • BASED_ON                 :   1
       • IS_EQUATION_IN           :   1
       • HAS_SOLUTION             :   1
       • HAS_COST_FOR_ADDITIONAL_UNIT:   1
       • HAS_ARGUMENT             :   1
       • HAS_PART                 :   1
       • IS_THE_RANGE             :   1
       • CALCULATED_OVER          :   1
       • HAS_ZERO_OVER            :   1
       • IS_DEFINED_WHEN          :   1
       • IS_CONTINUOUS_AT         :   1
       • IS_ONE_TO_ONE_ON         :   1
       • IS_EXPRESSED_AS          :   1

   🌐 GRAPH CONNECTIVITY
     Average Connections per Node: 1.30
     Isolated Nodes: 743
     Highly Connected Nodes (>5 connections): 58
     Top Connected Nodes:
       1. Node 3fa3ea84...: 61 connections
       2. Node 0421fc85...: 54 connections
       3. Node 9996fc32...: 51 connections
       4. Node 0a97b2ba...: 32 connections
       5. Node 32d33a79...: 28 connections

📋 SAMPLE ANALYSIS (Limited Search Results)
   Sample Relationship Types (8 found):
     • ASSESSED_BY         : 3
     • CAN_BE_EVALUATED_USING: 1
     • BASED_ON            : 1
     • CALCULATED_USING    : 1
     • PART_OF             : 1
     • EQUALS              : 1
     • IS_A                : 1
     • CONFIRMS            : 1

   Sample Entities:
     • The hyperbolic sine function, denoted as sinh(x), is a fundamental concept in hyperbolic geometry an...
     • A calculator utility is used to evaluate logarithmic expressions, specifically to find the approxima...
     • The symbol \(\)mathbb{R}\(\) represents the set of all real numbers. It is a fundamental concept in ...

   Sample Facts:
     • [CAN_BE_EVALUATED_USING] Use a calculating utility to evaluate [\log_3 7]
     • [BASED_ON] Use the composite function theorem.
     • [CALCULATED_USING] Calculate the limit using limit laws.

============================================================
