import random
import time
from RAG_MultiQuery import answer_question
import mesop as me
import mesop.labs as mel
from datetime import datetime
from typing import List, Dict, Any
from dataclasses import dataclass, field  # Import field from dataclasses

@me.stateclass
@dataclass
class State:
    sidenav_open: bool = False
    query_history: List[Dict[str, Any]] = field(default_factory=list)  # Use default_factory for mutable default

def on_click(e: me.ClickEvent):
    s = me.state(State)
    s.sidenav_open = not s.sidenav_open

SIDENAV_WIDTH = 200

@me.page(
    security_policy=me.SecurityPolicy(
        allowed_iframe_parents=["https://google.github.io", "https://huggingface.co"]
    ),
    path="/",
    title="Mesop Demo Chat",
)
def page():
    state = me.state(State)
    
    # Sidebar content
    with me.sidenav(
        opened=state.sidenav_open, style=me.Style(width=SIDENAV_WIDTH)
    ):
        me.text("History")
        if state.query_history:
            for entry in sorted(state.query_history, key=lambda x: x['timestamp']):
                timestamp = entry['timestamp']
                query = entry['input_query']
                response = entry['response']
                me.markdown(f"**{timestamp}**\n**Query:** {query}\n**Response:** {response}\n")
        else:
            me.text("No history available.")
    
    with me.box(
        style=me.Style(
            margin=me.Margin(left=SIDENAV_WIDTH if state.sidenav_open else 0),
        ),
    ):
        with me.content_button(on_click=on_click):
            me.icon("menu")
        mel.chat(transform, title="Maintenance assistant", bot_user="Agent")

def transform(input: str, history: list[mel.ChatMessage]):
    response = answer_question(input)
    
    # Get current timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Log query and response with timestamp
    s = me.state(State)
    s.query_history.append({
        'timestamp': timestamp,
        'input_query': input,
        'response': response
    })
    
    return response

if __name__ == '__main__':
    app = me.create_app()  # Assuming this is how Mesop app is created
    app.run(debug=True)
