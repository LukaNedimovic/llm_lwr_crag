import gradio as gr


class Page:
    """
    Base class for Gradio pages.
    """
    
    def render(self):
        """
        Define UI elements for the page.
        """
        raise NotImplementedError("Subclasses must implement render()")

    def update_ui(self, new_page):
        """
        Function to update the UI dynamically.
        """
        global current_page
        current_page = new_page
        gr.update()  # Trigger UI refresh