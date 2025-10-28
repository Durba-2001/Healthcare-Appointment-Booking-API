import asyncio
import json
from loguru import logger
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage
from mcp import ClientSession
from mcp.client.sse import sse_client
import os
from dotenv import load_dotenv
from src.utils.config import ACCESS_TOKEN, MCP_SERVER_URL, GOOGLE_API_KEY
from src.utils.helper_func import  extract_recommendation, extract_session
# --------------------------
# Load environment variables
# --------------------------
load_dotenv()

# --------------------------
# Initialize LLM
# --------------------------
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", api_key=GOOGLE_API_KEY)


class MCPClient:
    """Wrapper for communicating with the MCP server and selecting tools via LLM."""

    def __init__(self):
        self.server_url = MCP_SERVER_URL
        self.access_token = ACCESS_TOKEN
        self.conversations = {}

    async def process_user_message(self, chat_id: str, user_message: str, first_message: bool = False) -> dict:
        """Route user message through LLM -> MCP -> Tool Execution."""
        try:
            # -------------------------------
            # Initialize conversation session
            # -------------------------------
            if chat_id not in self.conversations:
                self.conversations[chat_id] = {
                    "last_tool": None,
                    "history": [],
                    "city": None,
                    "health_issue": None,
                    "selected_professional": None,
                    "name": None,
                    "age": None,
                    "contact": None,
                    "email": None
                }

            session_state = self.conversations[chat_id]

            # -------------------------------
            # Decide next tool via LLM
            # -------------------------------
            tool_name, args = await self._decide_tool_via_llm(chat_id, user_message)

           

            # -------------------------------
            # Call MCP tool via SSE
            # -------------------------------
            async with sse_client(self.server_url) as (read_stream, write_stream):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    logger.info(f"Executing tool: {tool_name}")
                    result = await session.call_tool(tool_name, arguments=args)

            # -------------------------------
            # Parse tool output
            # -------------------------------
            if hasattr(result, "content") and result.content:
                tool_output = getattr(result.content[0], "text", str(result))
            else:
                tool_output = str(result)

            logger.info(f"Tool Output:\n{tool_output}")

            # -------------------------------
            # Update session and history
            # -------------------------------
            session_state["last_tool"] = tool_name
            session_state["history"].append(
                {"user": user_message, "tool": tool_name, "output": tool_output}
            )

            # -------------------------------
            # Generate assistant summary
            # -------------------------------
            assistant_summary = await self._summarize_for_user(
                chat_id, user_message, tool_name, tool_output
            )

            return {
                "tool_used": tool_name,
                "response": tool_output,
                "assistant_summary": assistant_summary,
            }

        except Exception as e:
            logger.exception(f"MCPClient process_user_message error: {e}")
            return {
                "response": f"Error processing your request: {e}",
                "assistant_summary": "Sorry, I encountered a problem handling that message.",
            }

    # ----------------------------------------------------------------
    # LLM decides the next tool based on session & last conversation
    # ----------------------------------------------------------------
    async def _decide_tool_via_llm(self, chat_id: str, user_message: str):
        """Use Gemini to decide the next MCP tool based on conversation context."""

        if chat_id not in self.conversations:
            self.conversations[chat_id] = {
                "last_tool": None,
                "history": [],
                "city": None,
                "health_issue": None,
                "selected_professional": None,
                "name": None,
                "age": None,
                "contact": None,
                "email": None
            }

        session_state = self.conversations[chat_id]

        # Prepare recent conversation history
        history_text = "\n".join(
            f"User: {h['user']}\nTool({h['tool']})→ {h['output']}"
            for h in session_state["history"][-10:]
        )

        routing_prompt = f"""
You are an intelligent healthcare assistant routing system for an online appointment booking workflow.
You decide which tool (API endpoint) should be called next based on user messages, session data, and recent conversation history.

Your goal:
Guide the user step-by-step through the following sequence:
1. recommend_service(chat_id, user_message, token)
   → Understand the user's health concern and recommend a suitable medical service.
2. list_professionals(chat_id, user_message, token)
   → List available professionals for the chosen service and location.
3. select_professional(chat_id, user_message, token)
   → Capture which doctor or specialist the user chooses.
4. collect_user_info(chat_id, name, age, contact, email, token)
   → Ask for and record the user's personal details.
5. confirm_user_info(chat_id, user_message, token)
   → Confirm user details before proceeding.
6. check_availability(chat_id, user_message, token)
   → Check available time slots for booking.
7. confirm_booking(chat_id, user_message, token)
   → Confirm final booking only after explicit user consent.

Hard routing rules (apply in this order):
1. ALWAYS inspect session_state and the last 10 messages before choosing a tool.
2. If session_state indicates REQUIRED FIELDS are missing (any of: city, selected_professional, name, age, contact, email) → call **collect_user_info** (or list_professionals / select_professional if earlier steps are incomplete). Provide the user's raw message in user_message.
3. If the user message contains a clear personal-field update (case-insensitive keywords or patterns): "my name is", "name is", "age is", "i am \d{1,3}", "years old", "my phone", "phone is", "contact is", a 10-digit number, or contains "@" and "." (email) → call **collect_user_info** with user_message (do NOT call confirm_user_info).
4. Only call **confirm_user_info** when BOTH are true:
   a) session_state already contains **customer_details** (i.e., name/age/contact/email were stored or returned by the previous tool), AND
   b) the user's message is an explicit confirmation intent — e.g., exactly "yes", "y", "confirm", "ok", "no", "n", "wrong", or a short reply intended to accept/reject. In that case send the user's raw message as user_message to confirm_user_info.
5. If the previous step (check_availability) returned an available slot and asked for confirmation, and the user replies with confirmation intent → call **confirm_booking**.
6. If the user gives a date/time (e.g., contains a year or words like "at", "am", "pm", "tomorrow", or day+month) → call **check_availability** with user_message.
7. If the user is asking for list of doctors or named a city → call **list_professionals**.
8. Never skip steps. If previous step is incomplete, stay on that step.
9. If the LLM is unsure, default to the earliest incomplete step (recommend_service → list_professionals → select_professional → collect_user_info).
10. If the user types a date/time like "book on 2025-12-15 at 10:00", treat it as input for check_availability. - If a slot is unavailable, stay in check_availability until a valid slot is found and confirmed. 
11. If user provides unrelated or partial information, clarify or stay on the current tool. 
12. Be consistent and context-aware 
13. do not repeat completed steps unless necessary.

When deciding, use this logic order:
1 Read the full history (last 10 exchanges)
2 Check if current step is incomplete or user is still providing data
3 If yes → stay on same tool
4 If previous step confirmed → move to next tool
5 Return a structured JSON response with the tool name and arguments

Return **strictly** in this JSON format (no explanations, no comments):

{{
  "tool": "<tool_name>",
  "arguments": {{
    "chat_id": "{chat_id}",
    "user_message": "<processed message or confirmation prompt>"
  }}
}}

Known session info:
City: {session_state.get("city", "Not provided")}
Health issue: {session_state.get("health_issue", "Not provided")}
Selected professional: {session_state.get("selected_professional", "Not selected")}
User info: {session_state.get("name", "Not provided")}, {session_state.get("age", "Not provided")}, {session_state.get("contact", "Not provided")}, {session_state.get("email", "Not provided")}

Recent conversation history (for context and step tracking):
{history_text}

Current user message: "{user_message}"

Your task: Based on the entire conversation and rules above, choose the **correct tool** for the next step.
"""


        response = await llm.ainvoke([HumanMessage(content=routing_prompt)])
        decision_text = response.content.strip()

        # Clean possible markdown formatting
        if decision_text.startswith("```"):
            decision_text = decision_text.strip("`").replace("json", "", 1).strip()

        # Parse JSON
        try:
            parsed = json.loads(decision_text)
        except json.JSONDecodeError:
            logger.warning("Invalid JSON from LLM; defaulting to recommend_service.")
            # parsed = {"tool": "recommend_service", "arguments": {"user_message": user_message}}

        tool_name = parsed.get("tool", "recommend_service")
        args = parsed.get("arguments", {})
        args["chat_id"] = chat_id
        args["token"] = self.access_token

        return tool_name, args

    # ----------------------------------------------------------------
    # Generate short assistant summary
    # ----------------------------------------------------------------
    async def _summarize_for_user(self, chat_id: str, user_message: str, tool_name: str, tool_output: str) -> str:
        """Generate structured assistant response for each tool type."""

        tool_out = extract_recommendation(tool_output)

    # Special case: when user info is being collected
        if tool_name == "collect_user_info":
            summary_prompt = f"""
You are a polite healthcare assistant confirming user details.

The user message was:
"{user_message}"

The tool returned this output:
"{tool_out}"

Your job:
- Summarize what information (name, age, contact, email) has been recorded.
- Ask politely if these details are correct.
- If some details are missing or invalid, mention that and ask the user to provide them.
- Keep the message natural and short (1–2 sentences), suitable for chat display.
- Do not include JSON or technical text.
"""
        else:
        # Default summarization for other tools
            summary_prompt = f"""
You are a helpful healthcare assistant.

The user sent the message:
"{user_message}"

The tool returned this output:
"{tool_out}"

Your task:
- Extract only the recommendation text.
- Generate a clear, concise, natural-language reply for the user.
- Do not include any JSON, braces, quotes, or backslashes.
- Keep it short (1–2 sentences).
"""

        response = await llm.ainvoke([HumanMessage(content=summary_prompt)])
        return response.content.strip()
