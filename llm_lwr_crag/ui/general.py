import gradio as gr
from .page import Page


class URLInputPage(Page):
    def render(self):
        with gr.Column():
            gr.Markdown("## Enter a URL")
            url_box = gr.Textbox(label="Website URL")
            submit_button = gr.Button("Load")

        def load_url(url):
            """
            Switch to the next page when a URL is entered.
            """
            print("asd")
            return URLLoadedPage(url)

        submit_button.click(load_url, inputs=url_box, outputs=[])


class URLLoadedPage(Page):
    def __init__(self, url):
        self.url = url

    def render(self):
        with gr.Column():
            gr.Markdown(f"## URL Loaded: {self.url}")
            back_button = gr.Button("Go Back")

        def go_back():
            """Go back to the URL input page."""
            return URLInputPage()

        back_button.click(go_back, inputs=[], outputs=[])
        
        
