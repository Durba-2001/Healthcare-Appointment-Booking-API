import asyncio
import json
from loguru import logger
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage
from mcp import ClientSession
from mcp.client.sse import sse_client
import os
from dotenv import load_dotenv
from src.utils.config import ACCESS_TOKEN,MCP_SERVER_URL,GOOGLE_API_KEY
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
        self.server_url = MCP_SERVER_URL         # URL of the MCP Server
        self.access_token = ACCESS_TOKEN         # Token for authorized MCP communication
        self.conversations = {}                  # Stores conversation history per chat_id

    async def process_user_message(self, chat_id: str, user_message: str, first_message: bool = False) -> dict:
        """Route user message through LLM -> MCP -> Tool Execution."""
        try:
            # Create conversation state if new chat_id
            if chat_id not in self.conversations:
                self.conversations[chat_id] = {"last_tool": None, "history": []}

            session_state = self.conversations[chat_id]   # Get conversation history for this chat

            # Decide which tool to execute
            if first_message:
                # Force first tool to be recommend_service
                tool_name = "recommend_service"
                args = {"chat_id": chat_id, "user_message": user_message, "token": self.access_token}
            else:
                # Decide tool dynamically using LLM
                tool_name, args = await self._decide_tool_via_llm(chat_id, user_message)

            # Validate and clean arguments based on expected schema for chosen tool
            args = self._sanitize_tool_args(tool_name, args, user_message)

            # Connect to MCP server using SSE
            async with sse_client(self.server_url) as (read_stream, write_stream):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()                 # Initialize communication session
                    logger.info(f"Executing tool: {tool_name}")
                    result = await session.call_tool(tool_name, arguments=args)  # Call the selected tool

            # Retrieve the tool response text
            if hasattr(result, "content") and result.content:
                tool_output = getattr(result.content[0], "text", str(result))
            else:
                tool_output = str(result)

            logger.info(f"Tool Output:\n{tool_output}")

            # Save result to conversation history
            session_state["last_tool"] = tool_name
            session_state["history"].append({"user": user_message, "tool": tool_name, "output": tool_output})

            # Generate a short assistant summary to show the user
            assistant_summary = await self._summarize_for_user(chat_id, user_message, tool_name, tool_output)

            return {"tool_used": tool_name, "response": tool_output, "assistant_summary": assistant_summary}

        except Exception as e:
            logger.exception(f"MCPClient process_user_message error: {e}")
            return {
                "response": f"Error processing your request: {e}",
                "assistant_summary": "Sorry, I encountered a problem handling that message."
            }

    async def _decide_tool_via_llm(self, chat_id: str, user_message: str):
        """Use Gemini to decide which MCP tool to execute next."""
        conversation = self.conversations.get(chat_id, {"history": []})  # Get conversation history
        # Build last 5 messages to provide context to LLM
        history_text = "\n".join(
            f"User: {h['user']}\nTool({h['tool']})→ {h['output']}" for h in conversation["history"][-5:]
        )

        # Prompt LLM to choose the next tool
        routing_prompt = f"""
You are a healthcare assistant routing system.
Choose the correct MCP tool for the next step.

Available tools and expected args:
1. recommend_service(chat_id, user_message, token)
2. list_professionals(chat_id, user_message, token)
3. select_professional(chat_id, user_message, token)
4. collect_user_info(chat_id, name, age, contact, email, token)
5. confirm_user_info(chat_id, user_message, token)
6. check_availability(chat_id, user_message, token)
7. confirm_booking(chat_id, user_message, token)
Last tool should be confirm_booking.

Conversation so far:
{history_text}

User message: "{user_message}"
Return ONLY JSON with keys: tool, arguments.
"""

        # Get tool decision from LLM
        response = await llm.ainvoke([HumanMessage(content=routing_prompt)])
        decision_text = response.content.strip()

        # Remove code fences if present
        if decision_text.startswith("```"):
            decision_text = decision_text.strip("`").replace("json", "", 1).strip()

        # Parse JSON response from LLM
        try:
            parsed = json.loads(decision_text)
        except json.JSONDecodeError:
            logger.warning("Gemini returned invalid JSON; defaulting to recommend_service.")
            parsed = {"tool": "recommend_service", "arguments": {"user_message": user_message}}

        tool_name = parsed.get("tool", "recommend_service")  # Extract tool name
        args = parsed.get("arguments", {})                   # Extract tool arguments
        args["chat_id"] = chat_id                            # Add mandatory parameters
        args["token"] = self.access_token

        # Safety check: if user says "yes", do not send back to check_availability again
        if tool_name == "check_availability" and user_message.strip().lower() in ["yes", "y", "confirm", "ok"]:
            tool_name = "confirm_booking"

        return tool_name, args

    def _sanitize_tool_args(self, tool_name: str, args: dict, user_message: str) -> dict:
        """Ensure valid schema for each tool."""
        # All tools receive chat_id and token
        common = {"chat_id": args.get("chat_id"), "token": args.get("token")}

        # Special handling only for collect_user_info tool
        if tool_name == "collect_user_info":
            name = (args.get("name") or "").strip()        # Clean name string
            email = (args.get("email") or "").strip()      # Clean email string
            contact = args.get("contact") or args.get("phone_number") or ""  # Accept phone_number alias

            # Convert age to int if possible
            try:
                age = int(args.get("age")) if args.get("age") else None
            except (ValueError, TypeError):
                age = None

            return {**common, "name": name, "age": age, "contact": contact, "email": email}

        # Default schema for other tools: send user_message only
        return {**common, "user_message": user_message}

    async def _summarize_for_user(self, chat_id: str, user_message: str, tool_name: str, tool_output: str) -> str:
        """Generate 1–2 line assistant summary for display."""
        summary_prompt = f"""
Summarize in 2 lines what the assistant should reply next.
User message: "{user_message}"
Tool used: {tool_name}
Tool output: {tool_output}
"""
        response = await llm.ainvoke([HumanMessage(content=summary_prompt)])  # Ask LLM for summary
        return response.content.strip()                                       # Return clean summary