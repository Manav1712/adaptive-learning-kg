# Minimal Contracts (Inputs/Outputs)

## Coach (front door)
**Input**
```json
{ "student_id": "s_1", "session_id": "sess_1", "text": "What's vertex form?", "timestamp": "2025-08-12T10:32:00Z" }
```
**Output**
```json
{
  "action": "continue | branch | switch",
  "message_to_student": "Great—vertex form. We'll continue here.",
  "target_los": ["lo_vertex_form"],
  "center_node": "lo_vertex_form",
  "new_session_id": "sess_2"
}
```

... (Include all remaining agent specs in similar format from PDF) ...
