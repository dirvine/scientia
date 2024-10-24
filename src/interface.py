import gradio as gr

def create_interface(tokenizers, encode_text, show_encodings):
    """Create and launch the Gradio interface"""
    with gr.Blocks() as demo:
        gr.Markdown("# Text Encoder and Index Viewer")
        with gr.Tab("Encode Text"):
            gr.Markdown("## Encode Text with Different Tokenizers")
            gr.Markdown(
                "Enter text to encode using multiple tokenizers. "
                "The text will be tokenized and stored in vector databases for each tokenizer type."
            )
            text_input = gr.Textbox(label="Enter text to encode")
            encode_button = gr.Button("Encode")
            encode_output = gr.Textbox(label="Encoded Output")
            encode_button.click(encode_text, inputs=text_input, outputs=encode_output)
        with gr.Tab("Show Encodings"):
            gr.Markdown("## View Stored Encodings")
            gr.Markdown(
                "Select a tokenizer to view the stored vector encodings. "
                "This will show the contents of the vector database for the selected tokenizer."
            )
            tokenizer_dropdown = gr.Dropdown(choices=list(tokenizers.keys()), label="Select Tokenizer")
            show_button = gr.Button("Show Encodings")
            show_output = gr.Textbox(label="Index Contents")
            show_button.click(show_encodings, inputs=tokenizer_dropdown, outputs=show_output)
            gr.Markdown("### What are Encodings?")
            gr.Dropdown(
                choices=["Encodings Explanation"],
                label="Learn More",
                value="Encodings Explanation",
                interactive=False
            )
            gr.Markdown(
                "Encodings are numerical representations of text that capture semantic meaning. "
                "In large language models (LLMs), encodings are used to transform text into a format "
                "that the model can process. These encodings help the model understand context, relationships, "
                "and nuances in the text, enabling it to perform tasks like translation, summarization, and more."
            )

    demo.launch()
