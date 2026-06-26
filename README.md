# FinReport Agent – Coding Session I'm Most Proud Of

**Project:** FinReport Agent — Automated Financial & Compliance Report Generator  
**Tech:** LangChain + LangGraph + Model Context Protocol (MCP) + Groq + React

This session showcases the full multi-agent system I built. It uses a Star-shaped LangGraph architecture where a supervisor intelligently routes between specialized agents, with all tools isolated in a secure FastMCP server.

**User Prompt I Tested:**
"Analyze the latest quarterly financials for a SaaS company with $2.4M ARR, 35% MoM growth, and high churn in enterprise segment."

**What Happened in This Session:**

- The **Data Ingestion Agent** (via MCP tools) loaded and cleaned the uploaded data using pandas and SQLite.
- The **Analysis & Research Agents** (powered by Groq) processed financial ratios and generated insights.
- The **Compliance Agent** queried the ChromaDB RAG corpus and automatically flagged a minor disclosure gap I hadn’t explicitly prompted for, suggesting a fix.
- The **Risk Agent** calculated a clear risk score with detailed reasoning.
- The **Narrative/Report Agent** produced a polished, board-ready report.

The entire pipeline completed in **under 30 seconds** with almost no manual intervention.

**Why I’m Particularly Proud of This Session:**

Watching the agents collaborate smoothly through the LangGraph supervisor and MCP orchestration was incredibly satisfying. The compliance agent catching an edge case in real-time highlighted the power of this architecture. This session perfectly reflects what I enjoy most — exploring new technologies like MCP isolation for security and speed, LangGraph for reliable agent workflows, and seeing them come together into a practical, fast tool. It reinforced my passion for building intelligent, production-grade agentic systems.

**Full Project:**  
https://github.com/Akashshelke07/finreport-agent
