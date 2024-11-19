# app.py
import gradio as gr
from typing import List
from main import Orchestrator

def create_interface(orchestrator: Orchestrator) -> gr.Blocks:
    with gr.Blocks(title="AI Assistant") as demo:
        chatbot = gr.Chatbot(
            [],
            elem_id="chatbot",
            bubble_full_width=False,
            height=600
        )
        
        with gr.Row():
            msg = gr.Textbox(
                label="Your Message",
                placeholder="Type your message here...",
                scale=8
            )
            submit = gr.Button("Submit", scale=1)
        
        clear = gr.Button("Clear")

        async def process_message(message: str, history):
            response = await orchestrator.process_query(message, history)
            history.append((message, response))
            return "", history

        msg.submit(
            process_message,
            [msg, chatbot],
            [msg, chatbot]
        )
        
        submit.click(
            process_message,
            [msg, chatbot],
            [msg, chatbot]
        )
        
        clear.click(
            orchestrator.clear_context,
            None,
            [chatbot, msg]
        )

    return demo