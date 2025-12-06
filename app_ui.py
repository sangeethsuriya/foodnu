"""
Nutrition-Based Local Food Advisor App
Gradio Web UI
"""

import asyncio
import nest_asyncio
import gradio as gr

from config import AppConfig
from agents.advisor_agent import NutritionAdvisorAgent
from spoon_ai.chat import ChatBot

# Allow nested event loops (required for Gradio + async)
nest_asyncio.apply()

# Global agent instance
agent = None


def initialize_agent():
    """Initialize the nutrition advisor agent."""
    global agent
    if agent is None:
        print("Initializing Nutrition Advisor Agent...")
        print("Loading embedding model (this may take a moment on first run)...")
        agent = NutritionAdvisorAgent(
            llm=ChatBot(
                llm_provider=AppConfig.DEFAULT_LLM_PROVIDER,
                api_key=AppConfig.DEFAULT_LLM_API_KEY,
                model_name="gpt-4o-mini"
            )
        )
        print("[OK] Agent initialized!")
    return agent


def run_async(coro):
    """Run an async coroutine in a way that works with Gradio."""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)


def respond(message, chat_history):
    """Process a chat message and update history."""
    if not message or not message.strip():
        return "", chat_history or []
    
    # Ensure chat_history is a list
    if chat_history is None:
        chat_history = []
    
    try:
        agent = initialize_agent()
        # Run async agent in sync context
        response = run_async(agent.run(message))
        
        # Use tuple format (user_message, bot_response)
        chat_history.append((message, response))
        
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        print(f"Exception in respond: {error_msg}")
        import traceback
        traceback.print_exc()
        chat_history.append((message, error_msg))
    
    return "", chat_history


# Create Gradio interface
with gr.Blocks(title="Nutrition Advisor") as demo:
    gr.Markdown(
        """
        # 🥗 Nutrition-Based Food Advisor
        
        Ask me about nutrition, foods rich in specific vitamins, or get personalized dietary recommendations!
        """
    )
    
    chatbot = gr.Chatbot(height=400)
    msg = gr.Textbox(
        placeholder="Ask about nutrition... (e.g., 'What foods are rich in vitamin C?')",
        label="Your message",
        lines=1
    )
    
    with gr.Row():
        submit = gr.Button("Send", variant="primary")
        clear = gr.Button("Clear")
    
    gr.Examples(
        examples=[
            "What foods are rich in vitamin C?",
            "I need high protein vegetarian options",
            "Compare the nutrition of chicken vs tofu",
            "What are some iron-rich foods?",
        ],
        inputs=msg
    )
    
    # Event handlers
    msg.submit(respond, [msg, chatbot], [msg, chatbot])
    submit.click(respond, [msg, chatbot], [msg, chatbot])
    clear.click(lambda: ("", []), None, [msg, chatbot], queue=False)

if __name__ == "__main__":
    print("Starting Nutrition Advisor Web UI...")
    print(AppConfig.get_summary())
    print("\nInitializing agent on startup...")
    initialize_agent()  # Pre-initialize to avoid delay on first message
    print("\nLaunching web interface...")
    demo.launch(share=False, server_name="127.0.0.1", server_port=7860)
